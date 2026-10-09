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
