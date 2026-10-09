# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from aiohttp.test_utils import TestClient, TestServer
from videotrans.api.app import create_app
from videotrans.core.db import init_db, set_db_path, get_db_path


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    orig_path = get_db_path()
    test_db = tmp_path / "test_scanner.db"
    set_db_path(test_db)
    init_db()
    yield test_db
    set_db_path(orig_path)


@pytest.mark.asyncio
async def test_scan_folders_recursive_and_ignore_non_mp4(tmp_path):
    # Setup directories with mp4 and non-mp4 files
    dir_a = tmp_path / "folder_a"
    dir_b = tmp_path / "folder_b"
    sub_a = dir_a / "nested_sub"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    sub_a.mkdir(parents=True)

    file1 = dir_a / "video1.mp4"
    file2 = sub_a / "video2.MP4"
    file3 = dir_b / "video3.mp4"
    txt_file = dir_a / "readme.txt"
    other_file = sub_a / "sample.mkv"

    for f in [file1, file2, file3, txt_file, other_file]:
        f.write_bytes(b"mock video data " * 100)

    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # 1. Scan single root folder recursively
        res = await client.post("/api/media/scan-folders", json={
            "paths": [str(tmp_path)],
            "recursive": True,
        })
        assert res.status == 200
        data = await res.json()
        assert data["ok"] is True
        assert data["count"] == 3
        filenames = {v["filename"].lower() for v in data["videos"]}
        assert filenames == {"video1.mp4", "video2.mp4", "video3.mp4"}

        # 2. Scan multiple folders
        res_multi = await client.post("/api/media/scan-folders", json={
            "paths": [str(dir_a), str(dir_b)],
            "recursive": True,
        })
        assert res_multi.status == 200
        data_multi = await res_multi.json()
        assert data_multi["count"] == 3

        # 3. Non-recursive scan on dir_a
        res_non_rec = await client.post("/api/media/scan-folders", json={
            "paths": [str(dir_a)],
            "recursive": False,
        })
        assert res_non_rec.status == 200
        data_non_rec = await res_non_rec.json()
        assert data_non_rec["count"] == 1
        assert data_non_rec["videos"][0]["filename"] == "video1.mp4"
    finally:
        await client.close()
