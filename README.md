Files:
- `src/`
	- `supertypes.py`
		- Useful type wrappers to automate common patterns and extend Python's ordinary semantics.
	- `utilities.py`
		- Decorators, logging and debugging utilities, and other ease-of-life features.
	- `tools.py`
		- Classes for facilitating tool de/serialization and use, and a small collection of general-purpose tools.
 		- Allows for reading/writing tools to/from PNG images, like SillyTavern's character cards.
	- `completion.py`
		- A unified interface for text and chat completions with all OpenAI-compatible providers.
	- `server.py`
		- Runs an API for persistent access to a customizable list of completion methods.
	- `piping.py`
		- Utilities for building complex workflows.
	- `structure.py`
	- `datamgmt.py`

- `data/`
	- `providers.json`
		- A list of providers, along with urls and models. Used to determine which models are accessible.
	- `secrets.json`
		- A list of API keys and other secret values for LLM and tool access.
 		- Necessarily private; you can copy `secrets.example.json` and fill in the empty values.
	- `pipes/`
		- A folder for storing individual workflow steps (`pipes`) serialized as JSON objects.
 		- `html_transformer.json`
  			- A pipe for cleaning and formatting HTML code without changing the page content.
	- `caches/`
		- A folder for preserving information to be reused, such as scraped HTML data.
