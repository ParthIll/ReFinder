"""Deterministic extraction, persistence and API checks. No model downloads or OCR binary required."""
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from docx import Document
from pypdf import PdfReader, PdfWriter, Transformation

from evaluation.database import Cancelled, ChromaIndex, Filters
from refinder.extract import extract_pieces, visual_assets
from refinder.vision import VisualEmbedder, asset_path, capture_asset
from refinder_extractors.extractors import ocr


def word_data(text="", confidence=0):
    return {"text": [text], "conf": [confidence], "block_num": [1], "par_num": [1], "line_num": [1]}


def axis(index, dim=512):
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1
    return vector


class FakeVisual:
    def __init__(self, mapping, fail_after=None):
        self.mapping, self.calls, self.fail_after = mapping, [], fail_after

    def encode_assets(self, assets):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Simulated interruption")
        self.calls.append([a["digest"] for a in assets])
        return np.stack([self.mapping[a["digest"]] for a in assets])

    def encode_query(self, query):
        return axis(0)


class VisualPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"REFINDER_DATA_DIR": str(self.root / "appdata")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def picture(self, color="red", name="picture.png"):
        path = self.root / name
        Image.new("RGB", (400, 200), color).save(path)
        return path

    def asset(self, color="red", **location):
        with Image.new("RGB", (400, 200), color) as image:
            return capture_asset(image, {"image": 1, **location}, {"ocr_confidence": 0.0})

    def store(self):
        return ChromaIndex(self.root / "db")

    def pdf(self, picture, name="photo.pdf"):
        buffer = io.BytesIO()
        with Image.open(picture) as image:
            image.save(buffer, format="PDF")
        buffer.seek(0)
        writer = PdfWriter()
        page = writer.add_blank_page(width=700, height=1000)
        page.merge_transformed_page(PdfReader(buffer).pages[0], Transformation().translate(120, 350))
        path = self.root / name
        with path.open("wb") as file:
            writer.write(file)
        return path

    def test_high_confidence_ocr_stays_text_only(self):
        with patch.object(ocr.pytesseract, "image_to_data", return_value=word_data("Invoice", 98)) as read:
            pieces = extract_pieces(self.picture())
        self.assertEqual([p.text for p in pieces], ["Invoice"])
        self.assertEqual(visual_assets(pieces), [])
        self.assertEqual(read.call_count, 1)

    def test_low_confidence_keeps_usable_words_and_visual(self):
        data = {"text": ["Invoice", "noise"], "conf": [95, 10], "block_num": [1, 1],
                "par_num": [1, 1], "line_num": [1, 1]}
        with patch.object(ocr.pytesseract, "image_to_data", return_value=data):
            pieces = extract_pieces(self.picture())
        self.assertIn("Invoice", [p.text for p in pieces])
        self.assertEqual(len(visual_assets(pieces)), 1)
        self.assertEqual(visual_assets(pieces)[0]["ocr_confidence"], 52.5)

    def test_missing_ocr_binary_still_indexes_image(self):
        with patch.object(ocr.pytesseract, "image_to_data", side_effect=ocr.pytesseract.TesseractNotFoundError()):
            pieces = extract_pieces(self.picture())
        assets = visual_assets(pieces)
        self.assertEqual(len(assets), 1)
        self.assertTrue(asset_path(assets[0]).is_file())
        self.assertIn("TesseractNotFoundError", assets[0]["ocr_error"])

    def test_pdf_extracts_picture_without_document_margins(self):
        with patch.object(ocr.pytesseract, "image_to_data", return_value=word_data()):
            assets = visual_assets(extract_pieces(self.pdf(self.picture())))
        self.assertEqual(len(assets), 1)
        self.assertEqual((assets[0]["page"], assets[0]["image"]), (1, 1))
        with Image.open(asset_path(assets[0])) as image:
            self.assertEqual(image.size, (400, 200))  # PDF page is 700 x 1000

    def test_docx_preserves_native_text_and_includes_header_images(self):
        picture = self.picture()
        word = Document()
        word.add_paragraph("Wedding planning notes")
        word.add_picture(str(picture))
        word.add_picture(str(picture))  # repeated image should not run OCR twice
        word.sections[0].header.paragraphs[0].add_run().add_picture(str(self.picture("blue", "header.png")))
        path = self.root / "report.docx"
        word.save(path)
        with patch.object(ocr.pytesseract, "image_to_data", return_value=word_data()) as read:
            pieces = extract_pieces(path)
        self.assertIn("Wedding planning notes", [p.text for p in pieces])
        self.assertEqual(len(visual_assets(pieces)), 2)
        self.assertEqual(read.call_count, 2)
        with Image.open(asset_path(visual_assets(pieces)[0])) as image:
            self.assertEqual(image.size, (400, 200))

    def test_visual_assets_are_content_addressed(self):
        a, b = self.asset(page=1), self.asset(page=8)
        self.assertEqual(a["digest"], b["digest"])
        self.assertEqual(len(list(asset_path(a).parent.glob("*.png"))), 1)
        self.assertNotEqual(a["page"], b["page"])

    def test_transparent_images_use_document_background(self):
        with Image.new("RGBA", (200, 200), (0, 0, 0, 0)) as image:
            asset = capture_asset(image, {"image": 1}, {})
        with Image.open(asset_path(asset)) as result:
            self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))

    def test_model_vector_cache_reuses_duplicate_images(self):
        import torch

        encoder = VisualEmbedder.__new__(VisualEmbedder)
        encoder.device = "cpu"
        encoder.transform = lambda image: torch.zeros(3, 256, 256)
        encoder.model = Mock()
        encoder.model.encode_image.return_value = torch.tensor([axis(0).tolist()])
        asset = self.asset()
        first = encoder.encode_assets([asset, asset])
        second = encoder.encode_assets([asset])
        self.assertEqual(encoder.model.encode_image.call_count, 1)
        np.testing.assert_array_equal(first[0], second[0])

    def test_image_only_corpus_filters_and_parent_location(self):
        store = self.store()
        red, blue = self.asset(page=3), self.asset("blue")
        paths = [str(self.root / "a.pdf"), str(self.root / "b.docx")]
        generation = store.replace(self.root, paths, [[], []], "unused", visuals=[[red, red], [blue]])
        lock = threading.Lock()
        store.embed(None, generation, lock)
        encoder = FakeVisual({red["digest"]: axis(0), blue["digest"]: axis(1)})
        store.embed_visuals(lambda: encoder, generation, lock)
        self.assertTrue(store.embedding_complete())
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.visual_count(), 3)
        results = store.search("flower arrangement", None, visual_vec=axis(0))
        self.assertEqual([r["path"] for r in results], paths)
        self.assertEqual(results[0]["visual_match"], {"page": 3, "image": 1})
        self.assertEqual(results[0]["passage"], "")
        filtered = store.search("flower arrangement", None, filters=Filters(types={"docx"}), visual_vec=axis(0))
        self.assertEqual([r["path"] for r in filtered], paths[1:])

    def test_resume_after_interrupted_file(self):
        store = self.store()
        red, blue = self.asset(), self.asset("blue")
        generation = store.replace(self.root, [str(self.root / "a.pdf"), str(self.root / "b.pdf")],
                                   [[], []], "unused", visuals=[[red], [blue]])
        mapping = {red["digest"]: axis(0), blue["digest"]: axis(1)}
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            store.embed_visuals(lambda: FakeVisual(mapping, fail_after=1), generation, threading.Lock())
        restored = self.store()
        self.assertTrue(restored.load())
        self.assertEqual(restored.visual_embedded_docs, 1)
        encoder = FakeVisual(mapping)
        restored.embed_visuals(lambda: encoder, restored.generation, threading.Lock())
        self.assertEqual(encoder.calls, [[blue["digest"]]])
        self.assertEqual(restored.visual_count(), 2)

    def test_changed_deleted_added_and_cleared_images(self):
        store = self.store()
        red, blue = self.asset(), self.asset("blue")
        path = str(self.root / "a.pdf")
        generation = store.replace(self.root, [path], [[]], "unused", visuals=[[red]])
        encoder = FakeVisual({red["digest"]: axis(0), blue["digest"]: axis(1)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        generation = store.apply_changes([(0, path, [])], [0], visuals=[[blue]])
        self.assertEqual(store.visual_count(), 0)
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        self.assertEqual(store.visual_count(), 1)
        rows = store.visual_collection.get(include=["metadatas"])["metadatas"]
        self.assertEqual(rows[0]["digest"], blue["digest"])
        self.assertEqual(rows[0]["doc_id"], 1)
        store.clear()
        self.assertEqual(store.visual_count(), 0)
        self.assertFalse(store.ready())

    def test_stale_embedding_cannot_delete_new_vectors(self):
        store = self.store()
        asset = self.asset()
        first = store.replace(self.root, [str(self.root / "a.png")], [[]], "unused", visuals=[[asset]])
        generation = store.replace(self.root, [str(self.root / "b.png")], [[]], "unused", visuals=[[asset]])
        encoder = FakeVisual({asset["digest"]: axis(0)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        with self.assertRaises(Cancelled):
            store.embed_visuals(lambda: encoder, first, threading.Lock())
        with self.assertRaises(Cancelled):
            store.embed(None, first, threading.Lock())
        self.assertEqual(store.visual_count(), 1)

    def test_legacy_manifest_requests_one_time_image_reextraction(self):
        store = self.store()
        path = str(self.root / "old.pdf")
        store.replace(self.root, [path], [["Existing searchable text"]], "unused")
        data = json.loads(store.manifest_path.read_text())
        for key in ("visuals", "visual_embedded_docs", "visual_model"):
            data.pop(key, None)
        store.manifest_path.write_text(json.dumps(data))
        restored = self.store()
        self.assertTrue(restored.load())
        self.assertEqual(restored.file_text(path), "Existing searchable text")
        self.assertEqual(restored.stamp(0), (-1.0, -1))

    def test_literal_text_match_keeps_priority_over_visual_only_match(self):
        store = self.store()
        asset = self.asset()
        paths = [str(self.root / "notes.txt"), str(self.root / "photo.pdf")]
        generation = store.replace(self.root, paths, [["Flower arrangements for the wedding"], []],
                                   "unused", visuals=[[], [asset]])
        encoder = FakeVisual({asset["digest"]: axis(0)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        result = store.search('"flower arrangements"', None, visual_vec=axis(0))
        self.assertEqual(result[0]["path"], paths[0])
        self.assertIsNone(result[0]["visual_match"])
        self.assertIn("Flower arrangements", result[0]["passage"])
        self.assertEqual(store.keyword_search("wedding")[0]["path"], paths[0])

    def test_added_folder_reuses_old_vectors_and_removal_cleans_only_its_images(self):
        store = self.store()
        folder = self.root / "other"
        folder.mkdir()
        red, blue = self.asset(), self.asset("blue")
        path = str(self.root / "a.pdf")
        generation = store.replace(self.root, [path], [[]], "unused", visuals=[[red]])
        encoder = FakeVisual({red["digest"]: axis(0), blue["digest"]: axis(1)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        # Use a sibling root so it can be removed independently.
        sibling = self.root.parent / (self.root.name + "-other")
        generation = store.add(sibling, [str(sibling / "b.pdf")], [[]], visuals=[[blue]])
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        self.assertEqual(encoder.calls, [[red["digest"]], [blue["digest"]]])
        self.assertEqual(store.visual_count(), 2)
        store.remove_root(str(sibling))
        self.assertEqual(store.visual_count(), 1)
        self.assertEqual([r["path"] for r in store.search("photo", None, visual_vec=axis(0))], [path])

    def test_visual_more_like_this_excludes_source(self):
        store = self.store()
        asset = self.asset()
        paths = [str(self.root / "a.pdf"), str(self.root / "b.docx")]
        generation = store.replace(self.root, paths, [[], []], "unused", visuals=[[asset], [asset]])
        store.embed_visuals(lambda: FakeVisual({asset["digest"]: axis(0)}), generation, threading.Lock())
        result = store.similar(paths[0])
        self.assertEqual([r["path"] for r in result], paths[1:])
        self.assertIsNotNone(result[0]["visual_match"])
        self.assertEqual(result[0]["passage"], "")

    def test_two_modalities_do_not_double_a_documents_semantic_vote(self):
        signal = {"name": np.zeros(2), "n_words": 2, "phrase": set(), "keyword": [],
                  "dense": [(1, "unrelated text", 0)], "visual": [(0, {}), (1, {})]}
        score = ChromaIndex.fuse(signal, 2)
        self.assertEqual(score[0], score[1])
        self.assertAlmostEqual(score[0], 1/61)

    def test_api_visual_only_search_does_not_load_text_model(self):
        from fastapi.testclient import TestClient

        store = self.store()
        asset = self.asset(page=2)
        path = str(self.root / "match.pdf")
        generation = store.replace(self.root, [path], [[]], "unused", visuals=[[asset]])
        encoder = FakeVisual({asset["digest"]: axis(0)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        # Avoid opening the user's real persistent index when importing the API.
        with patch("evaluation.database.ChromaIndex", return_value=store), \
                patch.object(store, "load", return_value=False), \
                patch("evaluation.watcher.FolderWatcher", return_value=Mock()):
            server = importlib.import_module("evaluation.server")
        with patch.object(server, "store", store), patch.object(server, "visual_embedder", encoder), \
                patch.object(server, "_get_embedder", side_effect=AssertionError("Text model must stay unloaded")):
            client = TestClient(server.app)
            response = client.get("/search", params={"q": "flowers", "mode": "smart"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["results"][0]["path"], path)
            self.assertEqual(response.json()["results"][0]["visual_match"], {"page": 2, "image": 1})
            self.assertEqual(client.get("/status").json()["embedded_images"], 1)
            self.assertEqual(client.get("/search", params={"q": "flowers", "mode": "keyword"}).json()["results"], [])

    def test_visual_query_failure_keeps_text_search_working(self):
        from fastapi.testclient import TestClient

        store = self.store()
        asset = self.asset()
        path = str(self.root / "notes.txt")
        generation = store.replace(self.root, [path], [["Quarterly budget and payroll"]], "unused", visuals=[[asset]])
        encoder = FakeVisual({asset["digest"]: axis(0)})
        store.embed_visuals(lambda: encoder, generation, threading.Lock())
        with patch("evaluation.database.ChromaIndex", return_value=store), \
                patch.object(store, "load", return_value=False), \
                patch("evaluation.watcher.FolderWatcher", return_value=Mock()):
            server = importlib.import_module("evaluation.server")
        with patch.object(server, "store", store), \
                patch.object(server, "_get_visual_embedder", side_effect=RuntimeError("Model unavailable")):
            with self.assertLogs("refinder", level="WARNING"):
                response = TestClient(server.app).get("/search", params={"q": "quarterly budget"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["results"][0]["path"], path)


if __name__ == "__main__":
    unittest.main()
