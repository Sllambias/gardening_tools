import os
from typing import Optional, List


def subdirs(
    folder: str,
    join: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    """
    implementation by: https://github.com/MIC-DKFZ/batchgenerators
    """
    subdirectories = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if (
                entry.is_dir()
                and (prefix is None or entry.name.startswith(prefix))
                and (suffix is None or entry.name.endswith(suffix))
            ):
                dir_path = entry.path if join else entry.name
                subdirectories.append(dir_path)

    if sort:
        subdirectories.sort()
    return subdirectories


def subfiles(
    folder: str,
    join: bool = True,
    prefix: str = None,
    suffix: str = None,
    sort: bool = True,
) -> List[str]:
    if join:
        res = [
            os.path.join(folder, i)
            for i in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))
        ]
    else:
        res = [
            i
            for i in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))
        ]
    if sort:
        res.sort()
    return res
