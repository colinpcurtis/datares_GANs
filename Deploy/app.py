import base64
import os,sys
import datetime
from urllib.parse import quote as urlquote
import re
from flask import Flask, send_from_directory
import dash
from PIL import Image
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_extensions import Download
from dash.dependencies import Input, Output, State
from model import load_model, get_prediction
from setup import PROJECT_ROOT, IMG_DIR


UPLOAD_DIRECTORY = "uploaded_img"

# TODO: need to delete uploaded images after some time frame or we'll run out of space

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=[dbc.themes.MINTY])
app.title = 'CycleGan Demo'


# Frontend Components


# Nav Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Demo", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Github", href="https://github.com/colinpcurtis/datares_GANs"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="CycleGAN Image Style Transfer",
    brand_href="#",
    color="primary",
    dark=True,
)
# Inputbox
inputbox = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Div(id='input-image-upload'), 
                dbc.Row(
                [
                    dbc.Col( dcc.Upload(id="upload-image",
                                        children=dbc.Button("Upload", color="primary"),
                                        multiple=True), width="auto"),
                ],align="center"
            )
            ]
        )
    ],
    style={"width": "18rem","height":"18rem"}
)
# Processbox
processbox = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Row(
                [
                    dbc.Col( dbc.Button("Process", id="process", color="primary"), width="auto"),
                ],align="center"
            )
            ]
        )
    ],
    style={"width": "12rem","height":"12rem"}
)

# outputbox
outputbox = dbc.Card(
    [
        html.Div(id='output-image-upload'),
    ],
    style={"width": "18rem","height":"18rem"}
)

# app layout
app.layout = html.Div([
    navbar,
    html.H2('Interactive Image Translation',
            style={'font-weight': 'bold', 'padding-left': '10%', 'font-size': '120%'}),
    html.Div(
    [
        dbc.Row(
            [
                dbc.Col(inputbox, width="auto"),
                dbc.Col(processbox, width="auto"),
                dbc.Col(outputbox, width="auto"),
            ],
            style={'padding-left':'15%','padding-right':'15%','padding-top':'10%'},align="center"
        ),
    ]
)
]
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(IMG_DIR, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def parse_contents(image):
    encoded_image = base64.b64encode(open(image, 'rb').read())
    return html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '100%'})
    ])



# upload the img
@app.callback(Output('input-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),  prevent_initial_call=True)
def update_inputbox(list_of_contents, list_of_names):
    if list_of_names is not None and list_of_contents is not None:
        for name, data in zip(list_of_names, list_of_contents):
            save_file(name, data)

    current_img = os.path.join(IMG_DIR,list_of_names[0]) 

    encoded_image = base64.b64encode(open(current_img, 'rb').read())
    return html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '100%'})
    ])

# process the img
@app.callback(
    Output('output-image-upload', 'children'),
    [Input('process',"n_clicks")],
    State('upload-image', 'filename')
    
    )
def update_output(n, list_of_names):
    if n is not None:
        # process
        gen = load_model(f"{PROJECT_ROOT}/genB2A.pt")
        current_img = os.path.join(IMG_DIR,list_of_names[0]) 
        new_name = "new_" + list_of_names[0]
        pred = get_prediction(gen, current_img)
        pred.save(os.path.join(IMG_DIR,new_name))
        return parse_contents(os.path.join(IMG_DIR,new_name))


def uploaded_files():
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

def file_download_link(filename):
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8888)
