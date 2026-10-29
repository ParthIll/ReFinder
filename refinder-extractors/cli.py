"""CLI: refinder-index --folder ./Documents --exclude ./Documents/private --db files.db"""
import argparse
import json
from .indexer import index_folders


def main():
    parser = argparse.ArgumentParser(description='Index folders into a ReFinder SQLite database')
    parser.add_argument('--folder', action='append', required=True,
                        help='Folder to scan (repeatable)')
    parser.add_argument('--exclude', action='append', default=[],
                        help='Directory to skip, including descendants (repeatable)')
    parser.add_argument('--db', default='refinder.db', help='Output SQLite .db file')
    parser.add_argument('--force', action='store_true', help='Re-extract files even when unchanged')
    args = parser.parse_args()
    result = index_folders(args.folder, args.exclude, args.db, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
