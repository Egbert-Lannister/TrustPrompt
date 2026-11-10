"""Build the upload ZIP, validate using mysite, and render its actual template.

This never connects to the live database or publishes a project.
Run with D:\\Anaconda\\python.exe from any directory.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from html.parser import HTMLParser

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'support'
SITE = Path('D:/Python_Script/mysite')
ZIP = ROOT.parent / 'TrustPrompt-mysite-html-v1.zip'

class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.headings = []
        self.tags = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag)
        if tag == 'img':
            assert a.get('alt'), 'Image requires alt text'
            self.sources.append(a['src'])
        if tag == 'section': self.headings.append(a.get('class'))

source = (ROOT / 'index.html').read_text(encoding='utf-8')
doc = Document(); doc.feed(source)
assert doc.tags.count('html') == 1 and doc.tags.count('h2') == 4
assert all(c == 'project-html-section' for c in doc.headings)
assert not set(doc.tags) & {'script','style','iframe','svg','link'}
assert len(doc.sources) == len(set(doc.sources)) == 3
members = ['index.html', *doc.sources]
for member in members:
    p = (ROOT / member).resolve()
    assert p.is_relative_to(ROOT.resolve()) and p.is_file()
with zipfile.ZipFile(ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for member in members: z.write(ROOT / member, member)
with zipfile.ZipFile(ZIP) as z: assert z.testzip() is None

os.environ.update(DJANGO_SETTINGS_MODULE='mysite.settings', DATABASE_URL='sqlite:///:memory:',
                  SECRET_KEY='trustprompt-offline-package-validation', DEBUG='false')
sys.path.insert(0,str(SITE))
import django
django.setup()
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import connection
from django.template.loader import render_to_string
from django.test import override_settings
from django.utils import translation
from posts.html_import import parse_html_project_upload, sanitize_project_html, rewrite_html_asset_urls
from posts.models import Post

temporary_root = SUPPORT / '_validation_tmp'
temporary_root.mkdir(exist_ok=True)
tempfile.tempdir = str(temporary_root)
parsed = parse_html_project_upload(SimpleUploadedFile(ZIP.name,ZIP.read_bytes(),content_type='application/zip'))
try:
    assert not parsed['warnings'], parsed['warnings']
    assert len(parsed['assets']) == 3
    assert parsed['content'] == sanitize_project_html(parsed['content'])
    assert {i['section'] for i in parsed['assets']} == {'framework','results','comparison'}
    public_map = {name:'/media/trustprompt/'+Path(name).name for name in doc.sources}
    public_html = rewrite_html_asset_urls(parsed['content'],public_map)
    for local, public in public_map.items():
        assert f'src="{public}"' in public_html
        assert f'src="{local}"' not in public_html
    assert public_html == sanitize_project_html(public_html)
    project = Post(title=parsed['title'],description=parsed['description'],
                   date=datetime(2026,11,5,12,tzinfo=ZoneInfo('America/New_York')),
                   category='projects',project_status='active',project_content_mode='html',
                   project_tech_stack='Python, PyTorch, RGB-D, Depth Completion, Conformal Risk Control')
    context = dict(project=project,html=parsed['content'],is_html_project=True,
                   LANGUAGE_CODE='zh-hans',public_js_labels={},csrf_token='preview-only')
    def no_database(execute,sql,params,many,context):
        raise AssertionError('Preview must never access a database')
    with translation.override('zh-hans'), override_settings(TIME_ZONE='America/New_York', STORAGES={
        'default':{'BACKEND':'django.core.files.storage.FileSystemStorage'},
        'staticfiles':{'BACKEND':'django.contrib.staticfiles.storage.StaticFilesStorage'},
    }), connection.execute_wrapper(no_database):
        preview = render_to_string('project_detail.html',context)
    (SUPPORT/'preview.html').write_text(preview,encoding='utf-8')
    (SUPPORT/'sanitized-content.html').write_text(parsed['content'],encoding='utf-8')
    report = {
        'zip':str(ZIP),'zip_bytes':ZIP.stat().st_size,
        'sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),
        'title':parsed['title'],'description':parsed['description'],
        'members':[{'path':n,'bytes':(ROOT/n).stat().st_size,'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()} for n in members],
        'media':[{'title':a['title'],'section':a['section'],'group_name':a.get('group_name')} for a in parsed['assets']],
        'warnings':parsed['warnings'],
        'checks':{'zip_crc':True,'four_sections':True,'allowed_files_only':True,
                  'all_media_referenced':True,'real_mysite_importer':True,'sanitization_idempotent':True,
                  'media_url_rewriting':True,'real_project_template_rendered':True,'database_queries':0},
        'template_reference':str(SITE/'static/examples/project-page-template.html'),
        'content_sources':['docs/analysis_paper.md','docs/freeze_method_v1.md','docs/test_results.json',
                           'docs/holm_test_v1.md','docs/conformal_test.md','docs/tta_test.md'],
    }
    (SUPPORT/'validation-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
finally:
    staged = Path(parsed['staging_dir']).resolve()
    assert staged.is_relative_to(temporary_root.resolve()) and staged != temporary_root.resolve()
    shutil.rmtree(staged)

