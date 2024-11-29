"""
Set up package.
Required modules: pydantic_core, toolz (supertypes.py), exa_py, openai, PIL (tools.py), fastapi, httpx (server.py), tiktoken (completion.py)
"""
from setuptools import setup, find_packages

setup(
    name='tactics',
    version='0.1',
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
        'exa_py'
    ],
    entry_points={
        'console_scripts': [
            'tactics_api=tactics.api:main',  # allows running API as a script
        ],
    },
)
