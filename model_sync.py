import dropbox
import os
import contextlib
import time
import datetime
import sys


def sync_folder(dbx, folder, rootdir):
    if not os.path.exists(rootdir):
        print(rootdir, "does not exist on your filesystem")
        sys.exit(1)
    response = dbx.files_list_folder(folder, recursive=True)
    for md in response.entries:
        path = md.path_display
        name = path[path.rindex("/") :]
        local_path = os.path.join(rootdir, path.replace("/", os.path.sep)[1:])
        if name.find(".") == -1:
            if not os.path.exists(local_path):
                print(
                    "\033[38;2;255;255;0m",
                    local_path,
                    "directory has been made.",
                    "\033[0m",
                )
                os.makedirs(local_path)
            continue
        if os.path.exists(local_path):
            sync_file(dbx, local_path, md.path_display, md)
        else:
            print("\033[38;2;255;0;255m", "Downloading", path, "...", "\033[0m")
            res = download(dbx, path)
            with open(local_path, "wb") as file:
                file.write(res)
            print("\033[38;2;0;255;0m", path, "downloaded successfully.", "\033[0m")


def sync_file(dbx, local_path, path, md):
    mtime = os.path.getmtime(local_path)
    mtime_dt = datetime.datetime(*time.gmtime(mtime)[:6])
    size = os.path.getsize(local_path)
    if (
        isinstance(md, dropbox.files.FileMetadata)
        and mtime_dt >= md.client_modified
        and size == md.size
    ):
        print(
            "\033[38;2;169;169;169m", path, "is already synced [stats match]", "\033[0m"
        )
    else:
        print("Downloading", path, "...")
        res = download(dbx, path)
        with open(local_path, "wb") as file:
            file.write(res)
        print("\033[38;2;0;255;0m", path, "updated successfully.", "\033[0m")


def download(dbx, path):
    """Download a file.
    Return the bytes of the file, or None if it doesn't exist.
    """
    path = "/%s" % (path.replace(os.path.sep, "/"))
    while "//" in path:
        path = path.replace("//", "/")
    with stopwatch("download"):
        try:
            md, res = dbx.files_download(path)
        except dropbox.exceptions.HttpError as err:
            print("*** HTTP error", err)
            return None
    data = res.content
    return data


@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        # print('Total elapsed time for %s: %.3f' % (message, t1 - t0))


def sync_model():
    TOKEN = "5hyj23rAXg4AAAAAAAAAAV7tAtMh0LYb3gSy170WzO4W3ZDcrVALOPvkl9KmfNZw"
    dbx = dropbox.Dropbox(TOKEN)
    try:
        sync_folder(dbx, "/model_face", ".")
        sync_folder(dbx, "/model_mouth", ".")
    except:
        print("\033[38;2;255;0;0m", "model weights synchronization faild.", "\033[0m")
