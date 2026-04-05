import os
import shutil
class DirectoryHelper:
    @staticmethod
    def ResetFolder(folder_path):
        """Deletes and recreates a folder."""
        # Delete the folder if it exists
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        # Recreate the empty folder
        os.makedirs(folder_path)