# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, patch

import nbformat
import pytest
from jupyter_server_ydoc.loaders import FileLoader
from jupyter_server_ydoc.rooms import DocumentRoom
from jupyter_server_ydoc.test_utils import FakeContentsManager, FakeEventLogger, FakeFileIDManager
from jupyter_server_ydoc.utils import OutOfBandChanges
from jupyter_ydoc import YUnicode
from pycrdt import Decoder, Doc, Map


async def test_should_initialize_document_room_without_store(rtc_create_mock_document_room):
    content = "test"
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content)

    await room.initialize()
    assert room._document.source == content


async def test_should_initialize_document_room_from_store(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # TODO: We don't know for sure if it is taking the content from the store.
    # If the content from the store is different than the content from disk,
    # the room will initialize with the content from disk and overwrite the document

    id = "test-id"
    content = "test"
    store = await rtc_create_SQLite_store("file", id, content)
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    await room.initialize()
    assert room._document.source == content


async def test_should_overwrite_the_store(rtc_create_SQLite_store, rtc_create_mock_document_room):
    id = "test-id"
    content = "test"
    store = await rtc_create_SQLite_store("file", id, "whatever")
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    await room.initialize()
    assert room._document.source == content

    doc = YUnicode()
    await store.apply_updates(doc.ydoc)

    assert doc.source == content


def _decode_state_vector(state: bytes) -> dict[int, int]:
    decoder = Decoder(state)
    return {
        decoder.read_var_uint(): decoder.read_var_uint() for _ in range(decoder.read_var_uint())
    }


async def _wait_until_stored(store, key: str) -> Doc:
    for _ in range(100):
        stored_doc = Doc()
        await store.apply_updates(stored_doc)
        if key in stored_doc.get("state", type=Map):
            return stored_doc
        await asyncio.sleep(0.01)
    raise TimeoutError(f"{key!r} was never written to the store")


async def _create_store_with_session(rtc_create_SQLite_store, content: str):
    # A store with a document session, as written by any room since sessions exist; otherwise
    # the room backfills one with a full save, which would hide what these tests check.
    store = await rtc_create_SQLite_store("file", "test-id", content)
    previous_room = YUnicode()
    previous_room.ystate["document_session"] = "previous-session"
    await store.write(previous_room.ydoc.get_update())
    return store


async def test_store_keeps_writes_made_before_ready_when_loaded_from_store(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # Yjs (in the browser) holds back every item of a client whose clocks are not
    # contiguous, so the store must not miss the writes the room made before being ready.
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    async with room:
        await room.initialize()
        await room.ydoc_observed.wait()
        room._document.ystate["server_write"] = "after ready"
        stored_doc = await _wait_until_stored(store, "server_write")
    await loader.clean()

    room_clock = _decode_state_vector(room.ydoc.get_state())[room.ydoc.client_id]
    stored_clock = _decode_state_vector(stored_doc.get_state()).get(room.ydoc.client_id)
    assert stored_clock == room_clock


async def test_store_keeps_writes_made_while_loading_from_store(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # e.g. the file watcher updating the path while the store is being read
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)
    apply_updates = store.apply_updates

    async def apply_updates_with_concurrent_write(ydoc: Doc) -> None:
        room._document.ystate["during_load"] = "written while loading"
        await apply_updates(ydoc)

    with patch.object(store, "apply_updates", side_effect=apply_updates_with_concurrent_write):
        await room.initialize()

    stored_doc = Doc()
    await store.apply_updates(stored_doc)
    room_clock = _decode_state_vector(room.ydoc.get_state())[room.ydoc.client_id]
    stored_clock = _decode_state_vector(stored_doc.get_state()).get(room.ydoc.client_id)
    assert stored_clock == room_clock


async def test_store_keeps_writes_made_right_after_initialize(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    async with room:
        await room.initialize()
        room._document.ystate["server_write"] = "right after initialize"
        await _wait_until_stored(store, "server_write")
    await loader.clean()


async def test_store_keeps_writes_made_while_saving_to_store(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # e.g. the file watcher updating the path while the room saves its earlier writes
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)
    write = store.write

    async def write_with_concurrent_write(update: bytes) -> None:
        if "during_save" not in room._document.ystate:
            room._document.ystate["during_save"] = "written while saving"
        await write(update)

    async with room:
        with patch.object(store, "write", side_effect=write_with_concurrent_write):
            await room.initialize()
            stored_doc = await _wait_until_stored(store, "during_save")
    await loader.clean()

    room_clock = _decode_state_vector(room.ydoc.get_state())[room.ydoc.client_id]
    stored_clock = _decode_state_vector(stored_doc.get_state()).get(room.ydoc.client_id)
    assert stored_clock == room_clock


async def test_initialize_does_not_hang_when_room_tasks_cancelled_without_stop(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # e.g. the websocket server's task group cancelled at server shutdown
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)
    room.ydoc_observed_timeout = 0.1
    room_task = asyncio.create_task(room.start())
    await room.started.wait()
    await asyncio.sleep(0.01)
    room_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await room_task

    await asyncio.wait_for(room.initialize(), timeout=5)

    stored_doc = Doc()
    await store.apply_updates(stored_doc)
    assert room.ready
    assert _decode_state_vector(stored_doc.get_state()).get(room.ydoc.client_id)
    await loader.clean()


async def test_stale_dirty_flag_in_store_is_reset(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    previous_session = YUnicode(Doc(client_id=2**53 - 1))
    previous_session.dirty = True
    await store.write(previous_session.ydoc.get_update())
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    await room.initialize()

    stored_doc = Doc()
    await store.apply_updates(stored_doc)
    assert room._document.dirty is False
    assert stored_doc.get("state", type=Map).get("dirty") is False


async def test_failed_store_save_does_not_fail_open(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    with patch.object(store, "write", side_effect=OSError("disk I/O error")):
        await room.initialize()

    assert room.ready
    assert room._document.source == content


async def test_cancelled_progressive_open_releases_the_update_lock():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    loader = FileLoader(
        "test-id",
        FakeFileIDManager({"test-id": "test.ipynb"}),
        FakeContentsManager({"content": notebook, "writable": True}),
        poll_interval=0.1,
    )
    room = DocumentRoom(
        "test-room",
        "json",
        "notebook",
        loader,
        FakeEventLogger(),
        None,
        None,
        document_load_progressively=True,
    )

    with (
        patch.object(room, "_wait_for_ydoc_observed", side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await room.initialize()

    await asyncio.wait_for(room._update_lock.acquire(), timeout=5)
    room._update_lock.release()
    await loader.clean()


async def test_initialize_does_not_hang_when_room_not_started(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    await asyncio.wait_for(room.initialize(), timeout=5)

    assert room.ready


async def test_initialize_does_not_hang_when_room_stopped(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)
    room_task = asyncio.create_task(room.start())
    await room.started.wait()
    await asyncio.sleep(0.01)
    await room.stop()
    await asyncio.wait_for(room_task, timeout=5)

    await asyncio.wait_for(room.initialize(), timeout=5)

    assert room.ready
    await loader.clean()


async def test_stale_path_in_store_is_replaced_by_current_path(
    rtc_create_SQLite_store, rtc_create_mock_document_room
):
    # A file renamed while its room was closed keeps its history, with the old path.
    # The stale value is written by the highest possible client ID, so it would win
    # the conflict with the path set in the room constructor.
    content = "test"
    store = await _create_store_with_session(rtc_create_SQLite_store, content)
    renamed = YUnicode(Doc(client_id=2**53 - 1))
    renamed.path = "old.txt"
    await store.write(renamed.ydoc.get_update())
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, store=store)

    await room.initialize()

    stored_doc = Doc()
    await store.apply_updates(stored_doc)
    assert room._document.path == "test.txt"
    assert stored_doc.get("state", type=Map).get("path") == "test.txt"


async def test_defined_save_delay_should_save_content_after_document_change(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=0.01)

    await room.initialize()
    room._document.source = "Test 2"

    # Wait for a bit more than the poll_interval
    await asyncio.sleep(0.15)

    assert "save" in cm.actions


async def test_undefined_save_delay_should_not_save_content_after_document_change(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=None)

    await room.initialize()
    room._document.source = "Test 2"

    # Wait for a bit more than the poll_interval
    await asyncio.sleep(0.15)

    assert "save" not in cm.actions


async def test_should_not_save_content_when_all_clients_have_autosave_disabled(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=0.01)

    # Disable autosave for all existing clients
    for state in room.awareness._states.values():
        if state is not None:
            state["autosave"] = False

    # Inject a dummy client with autosave disabled
    room.awareness._states[9999] = {"autosave": False}

    await room.initialize()
    room._document.source = "Test 2"

    await asyncio.sleep(0.15)

    assert "save" not in cm.actions


async def test_should_save_content_when_at_least_one_client_has_autosave_enabled(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=0.01)

    # Disable autosave for all existing clients
    for state in room.awareness._states.values():
        if state is not None:
            state["autosave"] = False

    # Inject a dummy client with autosave enabled
    room.awareness._states[10000] = {"autosave": True}

    await room.initialize()
    room._document.source = "Test 2"

    await asyncio.sleep(0.15)

    assert "save" in cm.actions


async def test_manual_save_should_not_have_delay(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=0.5)

    await room.initialize()

    # Trigger a manual save
    room._save_to_disc()

    # Manual save should execute immediately, without waiting for the 0.5s delay
    # Check that save happens within a very short time (100ms should be enough)
    await asyncio.sleep(0.1)

    assert cm.actions.count("save") == 1


async def test_manual_save_with_pending_autosave_should_cancel_autosave(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=1.0)

    await room.initialize()

    room._document.source = "Test 2"

    await asyncio.sleep(0.1)

    assert cm.actions.count("save") == 0

    save_task = room._save_to_disc()

    # Manual save should execute immediately
    await asyncio.sleep(0.1)
    assert save_task.done()

    # Check that the manual save was recorded
    assert cm.actions.count("save") == 1

    await asyncio.sleep(1.0)

    # There should be only one save (the manual one), not two
    assert cm.actions.count("save") == 1


async def test_manual_save_should_execute_immediately_even_with_long_delay(
    rtc_create_mock_document_room,
):
    content = "test"
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=5.0)

    await room.initialize()

    save_task = room._save_to_disc()

    await asyncio.sleep(0.5)

    assert "save" in cm.actions
    assert save_task.done()


async def test_autosave_should_still_have_delay(
    rtc_create_mock_document_room,
):
    content = "test"
    save_delay = 0.3
    cm, _, room = rtc_create_mock_document_room(
        "test-id", "test.txt", content, save_delay=save_delay
    )

    await room.initialize()

    room._document.source = "Test 3"

    await asyncio.sleep(0.1)
    assert "save" not in cm.actions

    # Wait for the delay to complete
    await asyncio.sleep(save_delay)

    assert "save" in cm.actions


async def test_manual_save_should_work_when_save_delay_is_none_and_save_now_is_true(
    rtc_create_mock_document_room,
):
    """Test that manual saves execute even when save_delay is None."""
    content = "test"
    # When save_delay is None, autosave is disabled
    cm, _, room = rtc_create_mock_document_room("test-id", "test.txt", content, save_delay=None)

    await room.initialize()

    # Trigger a manual save with save_now=True
    # Even though save_delay is None, manual saves should still work
    await room._maybe_save_document(None, save_now=True)

    # Manual save should have executed
    assert cm.actions.count("save") == 1


# The following test should be restored when package versions are fixed.

# async def test_document_path(rtc_create_mock_document_room):
#     id = "test-id"
#     path = "test.txt"
#     new_path = "test2.txt"

#     _, loader, room = rtc_create_mock_document_room(id, path, "")

#     await room.initialize()
#     assert room._document.path == path

#     # Update the path
#     loader._file_id_manager.move(id, new_path)

#     # Wait for a bit more than the poll_interval
#     await asyncio.sleep(0.15)

#     assert room._document.path == new_path


async def test_on_outofband_change_skips_aset_when_content_unchanged(
    rtc_create_mock_document_room,
):
    """aset should not be called when out-of-band content matches the document."""
    content = "test"
    _, _, room = rtc_create_mock_document_room("test-id", "test.txt", content)
    await room.initialize()

    with patch.object(room._document, "aset", new_callable=AsyncMock) as mock_aset:
        await room._on_outofband_change()
        mock_aset.assert_not_called()

    assert not room._document.dirty


async def test_save_oob_skips_aset_when_content_unchanged(
    rtc_create_mock_document_room,
):
    """During save with OutOfBandChanges, aset should be skipped if content matches."""
    content = "test"
    cm, loader, room = rtc_create_mock_document_room(
        "test-id", "test.txt", content, save_delay=0.01
    )
    await room.initialize()

    with (
        patch.object(
            loader, "maybe_save_content", new_callable=AsyncMock, side_effect=OutOfBandChanges
        ),
        patch.object(room._document, "aset", new_callable=AsyncMock) as mock_aset,
    ):
        await room._maybe_save_document(None, save_now=True)
        mock_aset.assert_not_called()

    assert not room._document.dirty


async def test_override_content_raises_on_out_of_band_disk_change(
    rtc_create_mock_document_room,
):
    """override_content must not silently swallow a failed disk save.

    Unlike _maybe_save_document's autosave path (where an out-of-band disk
    change is expected and safe to adopt), an override is a client explicitly
    choosing to keep its own content -- silently falling back to whatever is
    on disk would defeat the point of the override. The caller
    (DocumentSessionOverrideHandler) needs a real error to surface instead of
    an override that looks like it succeeded while the disk write failed.
    """
    content = "test"
    _, loader, room = rtc_create_mock_document_room("test-id", "test.txt", content)
    await room.initialize()

    with (
        patch.object(
            loader, "maybe_save_content", new_callable=AsyncMock, side_effect=OutOfBandChanges
        ),
        pytest.raises(OutOfBandChanges),
    ):
        await room.override_content("client version")
