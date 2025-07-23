from flask import Blueprint, send_from_directory

bp_serve_uploads = Blueprint("serve_uploads", __name__)

@bp_serve_uploads.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('uploads', filename)
