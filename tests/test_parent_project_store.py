# -*- coding: utf-8 -*-
import pytest
from aiohttp.test_utils import TestClient, TestServer
from videotrans.api.app import create_app
from videotrans.core.db import init_db, set_db_path, get_db_path
from videotrans.core.project_store import (
    create_project,
    get_project,
    get_or_create_date_parent_project,
    list_subprojects,
    get_parent_project_dir,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    orig_path = get_db_path()
    test_db = tmp_path / "test_parent_projects.db"
    set_db_path(test_db)
    init_db()
    yield test_db
    set_db_path(orig_path)


def test_date_parent_project_and_subprojects():
    # 1. Create date parent project
    parent = get_or_create_date_parent_project("2026-10-09")
    assert parent["name"] == "2026-10-09"
    assert parent["state"]["is_parent"] is True

    # 2. Re-fetching on the same date returns the same parent project
    parent2 = get_or_create_date_parent_project("2026-10-09")
    assert parent2["id"] == parent["id"]

    # 3. Create subprojects
    sub1 = create_project(name="Video 1", parent_id=parent["id"])
    sub2 = create_project(name="Video 2", parent_id=parent["id"])
    assert sub1["parent_id"] == parent["id"]
    assert sub2["parent_id"] == parent["id"]

    # 4. List subprojects
    subs = list_subprojects(parent["id"])
    assert len(subs) == 2
    assert {s["name"] for s in subs} == {"Video 1", "Video 2"}


@pytest.mark.asyncio
async def test_batch_create_projects_endpoint():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        res = await client.post("/api/projects/batch-create", json={
            "date": "2026-10-09",
            "items": [
                {"name": "clip1.mp4", "duration": 12.5},
                {"name": "clip2.mp4", "duration": 34.0},
            ]
        })
        assert res.status == 201
        data = await res.json()
        assert data["ok"] is True
        assert data["parent"]["name"] == "2026-10-09"
        assert len(data["projects"]) == 2
        p1 = data["projects"][0]
        assert p1["parent_id"] == data["parent"]["id"]
        assert p1["name"] == "clip1.mp4"

        # Verify listing subprojects endpoint
        res_sub = await client.get(f"/api/projects/{data['parent']['id']}/subprojects")
        assert res_sub.status == 200
        sub_data = await res_sub.json()
        assert len(sub_data["subprojects"]) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_parenting_and_today_parent_endpoint():
    app = create_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # 1. Fetch today's parent
        res_today = await client.get("/api/projects/today-parent")
        assert res_today.status == 200
        today_data = await res_today.json()
        parent_id = today_data["parent"]["id"]
        assert "parent_" in parent_id
        assert today_data["parent"]["state"]["is_parent"] is True

        # 2. Create project without parentId -> auto-assigned to today's parent
        res_create = await client.post("/api/projects", json={
            "name": "Single Upload Video.mp4",
            "duration": 50.0,
        })
        assert res_create.status == 201
        created = await res_create.json()
        assert created["parent_id"] == parent_id

        # 3. Verify subprojects list on today's parent includes the new video
        res_subs = await client.get(f"/api/projects/{parent_id}/subprojects")
        assert res_subs.status == 200
        subs_data = await res_subs.json()
        assert any(s["id"] == created["id"] for s in subs_data["subprojects"])

        # 4. Verify cascading delete
        from videotrans.core.project_store import delete_project, get_project
        delete_project(parent_id)
        assert get_project(parent_id) is None
        assert get_project(created["id"]) is None
    finally:
        await client.close()


def test_subprojects_have_isolated_target_dirs_and_do_not_share_srt(tmp_path):
    from pathlib import Path
    from videotrans.core.project_store import (
        get_or_create_date_parent_project,
        create_project,
        get_project_dir,
    )
    from videotrans.api.task_params import build_task_params
    from videotrans.task.taskcfg import InputFile

    parent = get_or_create_date_parent_project("2026-10-12")
    sub1 = create_project(name="Video 1.mp4", parent_id=parent["id"])
    sub2 = create_project(name="Video 2.mp4", parent_id=parent["id"])

    dummy_video1 = tmp_path / "video1.mp4"
    dummy_video1.write_bytes(b"dummy video 1 content")
    dummy_video2 = tmp_path / "video2.mp4"
    dummy_video2.write_bytes(b"dummy video 2 content")

    def mock_format_video(p):
        path = Path(p)
        return InputFile(
            name=str(path),
            dirname=str(path.parent),
            basename=path.name,
            noextname=path.stem,
            ext=path.suffix.lstrip("."),
            uuid=f"uuid-{path.stem}",
        )

    base_options = {
        "recognType": 0,
        "sourceLanguage": "zh-cn",
        "targetLanguage": "vi",
    }

    params1 = build_task_params(dummy_video1, base_options, job_type="asr", project_id=sub1["id"], format_video_fn=mock_format_video)
    params2 = build_task_params(dummy_video2, base_options, job_type="asr", project_id=sub2["id"], format_video_fn=mock_format_video)

    # 1. target_dir must be isolated per subproject, not shared parent folder
    assert params1["target_dir"] != params2["target_dir"]
    sub1_dir = get_project_dir(sub1["id"])
    sub2_dir = get_project_dir(sub2["id"])
    assert Path(params1["target_dir"]).is_relative_to(sub1_dir)
    assert Path(params2["target_dir"]).is_relative_to(sub2_dir)

    # 2. Placing an SRT in subproject 1's target directory does not leak to subproject 2
    sub1_target = Path(params1["target_dir"])
    sub2_target = Path(params2["target_dir"])
    sub1_srt = sub1_target / "zh-cn.srt"
    sub1_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nSub 1 subtitle\n", encoding="utf-8")

    sub2_srt = sub2_target / "zh-cn.srt"
    assert not sub2_srt.exists()


