import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, request, redirect, session, url_for
import json
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Spotify API credentials
client_id = 'f5ba4e49430343c080fb07ba186566ce'
client_secret = 'd4177ed703e44a1ba73d339fd57662d9'
redirect_uri = 'http://localhost:5000/callback'

# Scope for accessing user's playlists
scope = 'playlist-read-private'

sp_oauth = SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope)

@app.route('/')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('playlists'))

@app.route('/playlists')
def playlists():
    token_info = session.get('token_info', None)
    if not token_info:
        return redirect(url_for('login'))

    sp = spotipy.Spotify(auth=token_info['access_token'])

    # Get the user's playlists
    results = sp.current_user_playlists(limit=1)
    playlists = results['items']

    # Collect track features and labels
    features_list = []
    labels_list = []

    for playlist in playlists:
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        tracks = sp.playlist_tracks(playlist_id)
        track_ids = [track['track']['id'] for track in tracks['items'] if track['track']]
        features = get_tracks_features(sp, track_ids)
        features_list.extend(features)
        labels_list.extend([playlist_name])  # Append playlist name once for all tracks in the playlist

    # Filter out None values
    features_list = [f for f in features_list if f]
    labels_list = labels_list[:len(features_list)]  # Ensure labels list has same length as features list

    # Save features and labels to JSON files
    with open('features.json', 'w') as f:
        json.dump(features_list, f)
    with open('labels.json', 'w') as f:
        json.dump(labels_list, f)

    return 'Data collection complete!'


def get_track_features(sp, track_id):
    features = sp.audio_features(track_id)
    return features[0] if features else None

def get_tracks_features(sp, track_ids):
    return [get_track_features(sp, track_id) for track_id in track_ids]


if __name__ == '__main__':
    app.run(debug=True)
