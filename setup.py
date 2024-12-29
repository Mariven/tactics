"""
Set up package.
Required modules: pydantic_core, pydantic, toolz (supertypes.py), exa_py, openai, PIL (tools.py), fastapi, httpx (server.py), tiktoken (completion.py), requests (utilities.py)
"""
from setuptools import setup, find_packages

setup(
    name='tactics',
    version='0.3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,  # ensure that JSON and other files are included
    install_requires=[
        'pydantic_core',
        'pydantic',
        'PIL',
        'fastapi',
        'httpx',
        'tiktoken',
        'openai',
        'exa_py',
        'toolz',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'tactics_server=tactics.server:main',  # allows running API as a script
        ],
    },
)
