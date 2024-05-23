from flask import Flask, redirect, session, url_for, request, jsonify
import os
import json
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import spotipy

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI')
scope = 'user-library-read playlist-read-private'

sp_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)


if __name__ == '__main__':
    app.run(debug=True)



