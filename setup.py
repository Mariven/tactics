"""
Set up package.
Required modules: pydantic_core (supertypes.py), exa_py, openai, PIL (tools.py), fastapi, httpx (api.py), tiktoken (completion.py)
"""
from setuptools import setup, find_packages

setup(
    name='noema',
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
            'noema_api=noema.api:main',  # allows running API as a script
        ],
    },
)
