import base64
import os
import datetime
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
from PIL import Image
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

UPLOAD_DIRECTORY = "images/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)
app.title = 'Black and White Converter'


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = html.Div([
    html.H2('Created by Adhvaith Vijay',
            style={'font-weight': 'bold', 'padding-left': '4px', 'font-size': '120%'}),
    dcc.Upload(
        id="upload-image",
        children=html.Div(
            ["Drag and drop or click to select a file to upload."]
        ),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        multiple=True,
    ),
    html.Div(id='output-image-upload')
    # html.H1("File Browser"),
    # html.H2("Upload"),
    # dcc.Upload(
    #     id="upload-data",
    #     children=html.Div(
    #         ["Drag and drop or click to select a file to upload."]
    #     ),
    #     style={
    #         "width": "100%",
    #         "height": "60px",
    #         "lineHeight": "60px",
    #         "borderWidth": "1px",
    #         "borderStyle": "dashed",
    #         "borderRadius": "5px",
    #         "textAlign": "center",
    #         "margin": "10px",
    #     },
    #     multiple=True,
    # ),
    # html.H2("File List"),
    # html.Ul(id="file-list"),
],
    style={"max-width": "500px"}
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def parse_contents(image):
    encoded_image = base64.b64encode(open(image, 'rb').read())
    return html.Div([
        # html.H5(filename),
        # html.H6(datetime.datetime.fromtimestamp(date)),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '100%'})
        # html.Img(src=contents)
        # html.Hr(),
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'), prevent_initial_call=True)
def update_output(list_of_contents, list_of_names):
    if list_of_names is not None and list_of_contents is not None:
        for name, data in zip(list_of_names, list_of_contents):
            save_file(name, data)

    image_file = Image.open('images/{}'.format(list_of_names[0]))  # open colour image
    image_file = image_file.convert('1')  # convert image to black and white
    image_file.save('result.png')

    return parse_contents('result.png')


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
