"""Negative controls for the release boundary, independent of private research data."""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest
import check_release
import release_lib


def archive(path, name='output.bin', payload=b'original output', kind=None):
    with tarfile.open(path, 'w:gz') as out:
        item = tarfile.TarInfo(name)
        if kind == 'symlink':
            item.type = tarfile.SYMTYPE
            item.linkname = '/outside'
        else:
            item.size = len(payload)
        out.addfile(item, io.BytesIO(payload) if kind is None else None)


def test_policy_rejects_game_data_secrets_and_absolute_links(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/'fake.wad').write_bytes(b'IWAD')
    (tmp_path/'absolute-link').symlink_to('/outside')
    (tmp_path/'credential.txt').write_text('ghp_' + 'x'*40)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    errors = check_release.scan_tree(tmp_path)
    assert any('fake.wad' in error and 'game data' in error for error in errors)
    assert any('absolute-link' in error and 'absolute symlink' in error for error in errors)
    assert any('credential.txt' in error and 'likely secret' in error for error in errors)


@pytest.mark.parametrize('name,kind', [('../escape', None), ('/absolute', None), ('C:/escape', None),
                                      (r'..\escape', None), ('link', 'symlink')])
def test_unsafe_archive_members_are_rejected(tmp_path, name, kind):
    bundle = tmp_path/'bad.tar.gz'
    archive(bundle, name=name, kind=kind)
    with pytest.raises(ValueError, match='Unsafe'):
        check_release.scan_bundle(bundle)


def test_bundle_policy_rejects_wad(tmp_path):
    bundle = tmp_path/'bad.tar.gz'
    archive(bundle, name='fake.wad', payload=b'IWAD')
    assert any('game data' in error for error in check_release.scan_bundle(bundle))


def test_authenticated_archive_still_requires_every_file_digest(tmp_path, monkeypatch):
    bundle = tmp_path/'bundle.tar.gz'
    payload = b'original output'
    archive(bundle, payload=payload)
    manifest = tmp_path/'manifest.json'
    expected = dict(archive_sha256=release_lib.sha(bundle),
                    files={'output.bin': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
    manifest.write_text(json.dumps(expected))
    monkeypatch.setattr(release_lib, 'MANIFEST', manifest)
    release_lib.unpack_verified(bundle, tmp_path/'accepted')
    assert (tmp_path/'accepted/output.bin').read_bytes() == payload
    # One changed output byte fails even when the archive envelope is rehashed.
    archive(bundle, payload=b'X'+payload[1:])
    expected['archive_sha256'] = release_lib.sha(bundle)
    manifest.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match='entry SHA-256 mismatch'):
        release_lib.unpack_verified(bundle, tmp_path/'rejected')
    # A changed archive with the original trusted envelope hash fails before extraction.
    expected['archive_sha256'] = '0'*64
    manifest.write_text(json.dumps(expected))
    with pytest.raises(ValueError, match='Bundle SHA-256 differs'):
        release_lib.unpack_verified(bundle, tmp_path/'not-extracted')
    assert not (tmp_path/'not-extracted').exists()


def test_metadata_privacy_changes_only_home_paths():
    from build_release_bundle import public_metadata
    original = dict(elf_sha256='a'*64, drom_sha256='b'*64, count=7,
                    commands=['/Users/sample/compiler', r'C:\Users\sample\compiler'],
                    error="KeyError('final_registers')")
    raw, count = public_metadata(json.dumps(original).encode())
    result = json.loads(raw)
    assert count == 2
    assert result['commands'] == ['/REDACTED_HOME/compiler', r'/REDACTED_HOME\compiler']
    assert {k: v for k, v in result.items() if k != 'commands'} == {
        k: v for k, v in original.items() if k != 'commands'}
