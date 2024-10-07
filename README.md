src/
	supertypes.py
		Useful type wrappers to automate common patterns and extend Python's ordinary semantics
	utilities.py
		Decorators, logging and debugging utilities, and other ease-of-life features
	tool_calls.py
		Classes for facilitating tool (de)serialization and use, and a small collection of tools
	structure.py
		A unified interface for text and chat completions with all OpenAI-compatible providers
	api.py
		An API for persistent access to a customizable list of completion methods
		
data/
	providers.json
		A list of providers, along with urls and models. Used to determine which models are accessible.
	secrets.json
		A list of API and other keys. Necessarily private; you can copy secrets.example.json and fill in the empty values.

