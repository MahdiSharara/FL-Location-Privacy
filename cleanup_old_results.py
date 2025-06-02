#!/usr/bin/env python
"""
cleanup_old_results.py - Archive old results directories to reduce workspace clutter.

This script moves old results directories to an 'archived_results' folder to keep the
workspace clean while preserving important experimental data.
"""

import os
import shutil
import glob
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_results(workspace_path=None, keep_recent=2):
    """
    Archive old results directories to keep workspace clean.
    
    Args:
        workspace_path: Path to the workspace. If None, uses current directory.
        keep_recent: Number of most recent results directories to keep in main folder.
    """
    if workspace_path is None:
        workspace_path = os.path.dirname(os.path.abspath(__file__))
    
    # Find all results directories
    results_pattern = os.path.join(workspace_path, "results_*")
    results_dirs = glob.glob(results_pattern)
    
    if not results_dirs:
        logger.info("No old results directories found.")
        return
    
    # Sort by modification time (most recent first)
    results_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    logger.info(f"Found {len(results_dirs)} results directories")
    
    # Create archive directory
    archive_dir = os.path.join(workspace_path, "archived_results")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Keep the most recent directories, archive the rest
    dirs_to_archive = results_dirs[keep_recent:]
    dirs_to_keep = results_dirs[:keep_recent]
    
    logger.info(f"Keeping {len(dirs_to_keep)} most recent directories:")
    for dir_path in dirs_to_keep:
        logger.info(f"  - {os.path.basename(dir_path)}")
    
    if dirs_to_archive:
        logger.info(f"Archiving {len(dirs_to_archive)} older directories:")
        
        for dir_path in dirs_to_archive:
            dir_name = os.path.basename(dir_path)
            archive_path = os.path.join(archive_dir, dir_name)
            
            try:
                # Move directory to archive
                shutil.move(dir_path, archive_path)
                logger.info(f"  - Archived {dir_name}")
            except Exception as e:
                logger.error(f"  - Failed to archive {dir_name}: {e}")
    
    # Create archive summary
    summary_path = os.path.join(archive_dir, "archive_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Results Archive Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Archive created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total archived directories: {len(dirs_to_archive)}\n\n")
        f.write("Archived directories:\n")
        for dir_path in dirs_to_archive:
            f.write(f"  - {os.path.basename(dir_path)}\n")
        f.write(f"\nKept in main workspace:\n")
        for dir_path in dirs_to_keep:
            f.write(f"  - {os.path.basename(dir_path)}\n")
    
    logger.info(f"Archive summary saved to: {summary_path}")
    logger.info("Cleanup completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Archive old results directories')
    parser.add_argument('--keep', type=int, default=2, 
                        help='Number of recent directories to keep (default: 2)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Workspace path (default: current directory)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be archived without actually moving files')
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Just show what would be archived
        workspace_path = args.workspace or os.path.dirname(os.path.abspath(__file__))
        results_pattern = os.path.join(workspace_path, "results_*")
        results_dirs = glob.glob(results_pattern)
        results_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        print(f"Found {len(results_dirs)} results directories:")
        print(f"Would keep {args.keep} most recent:")
        for i, dir_path in enumerate(results_dirs):
            status = "KEEP" if i < args.keep else "ARCHIVE"
            print(f"  {status}: {os.path.basename(dir_path)}")
    else:
        cleanup_old_results(args.workspace, args.keep)
