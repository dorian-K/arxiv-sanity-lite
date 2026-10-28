
# I run this to update the database with newest papers every day or so or etc.
up:
	uv run python arxiv_daemon.py --num 2000
	uv run python compute.py

# I use this to run the server
fun:
	uv run flask --app serve run
