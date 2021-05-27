import base64
import os, sys
import datetime
from urllib.parse import quote as urlquote
import re
from flask import Flask, send_from_directory
import dash
import io
import base64
from PIL import Image
import torchvision.transforms as transforms
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from model import load_model, get_prediction
from setup import PROJECT_ROOT, IMG_DIR
from dash_extensions import Download
from dash_extensions.snippets import send_file

UPLOAD_DIRECTORY = "uploaded_img"
gen = load_model(f"{PROJECT_ROOT}/genB2A.pt")
# load model at server startup so we don't waste time loading it when we get an inference request

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=[dbc.themes.MINTY])
app.title = 'Make-A-Monet'
IMAGE_SIZE = 512

# Frontend Components
dropdown_bar = dbc.Row(
    [dbc.DropdownMenu(
            direction="left",
            children=[
                # dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Home", href="/"),
                dbc.DropdownMenuItem("GitHub", href="https://github.com/colinpcurtis/datares_GANs"),
                dbc.DropdownMenuItem("Model Architecture", href="/page-2"),
            ],
            label="More"
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center"
)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("UCLA DataRes Spring 2021: Make-A-Monet", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            )
        ),
        dbc.Collapse(dropdown_bar, navbar=True),
    ],
    color="primary",
    dark=True,
)


# Inputbox
inputbox = html.Div([
    html.H4("Original Image", style={'font-weight': 'bold'}),
    dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(id='input-image-upload')
                ]
            )
        ],
        style={"width": "23rem", "height": "23rem"}),
    dcc.Upload(id="upload-image",
               children=dbc.Button("Upload", color="primary", size='lg'),
               multiple=True)
])

processbox = dbc.Row(
    [
        dbc.Col(dbc.Button("Process", id="process", color="primary", size='lg'), width="auto"),
    ], align="center"
)

outputbox = html.Div([
    html.H4("Transformed Image", style={'font-weight': 'bold'}),
    dbc.Card(
        [
            dbc.CardBody(
                [
                    dcc.Loading(children=[html.Div(id='output-image-upload')], color = "#199DFF", type="dot", fullscreen=True),
                    # html.Div(id='output-image-upload')
                ]
            ),
        ],
        style={"width": "23rem", "height": "23rem"}),
    html.Div([dbc.Button("Download", id="download-btn", color="primary", size='lg'), Download(id="download")])
])

my_content = html.Div(id="page-content", children=[])
# app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    my_content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/page-2":
        image_filename = 'cycle_gan_architecture.png'
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return [
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '68%'})
            ], style={'textAlign': 'center'})
        ]
    elif pathname == "/":
        return [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(inputbox, width="auto"),
                            dbc.Col(processbox, width="auto", style={'padding-bottom': '3%'}),
                            dbc.Col(outputbox, width="auto"),
                        ],
                        style={'padding-left': '5%', 'padding-right': '15%', 'padding-top': '5%'}, align="center"
                    ),
                ]
            )
        ]


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
              State('upload-image', 'filename'), prevent_initial_call=True)
def update_inputbox(list_of_contents, list_of_names):
    if list_of_names is not None and list_of_contents is not None:
        for name, data in zip(list_of_names, list_of_contents):
            save_file(name, data)

    current_img = os.path.join(IMG_DIR, list_of_names[0])
    encoded_image = base64.b64encode(open(current_img, 'rb').read())
    imgdata = base64.b64decode(encoded_image)
    img = Image.open(io.BytesIO(imgdata))
    # pt_centercrop_transform_rectangle = transforms.CenterCrop(512)
    # centercrop_rectangle = pt_centercrop_transform_rectangle(img)
    transform_list = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                         transforms.CenterCrop(IMAGE_SIZE)])
    res = transform_list(img)
    print(res.size)

    return html.Div([
        html.Img(src=res, style={'width': '100%', 'height': "20.5rem"})
    ])

# process the img
@app.callback(
    Output('output-image-upload', 'children'),
    [Input('process', "n_clicks")],
    State('upload-image', 'filename'), prevent_initial_call=True)
def update_output(n, list_of_names):
    if n is not None:
        # process
        current_img = os.path.join(IMG_DIR, list_of_names[0])
        new_name = "new_" + list_of_names[0]
        pred = get_prediction(gen, current_img)
        print(pred.size)
        # remove the original uploaded image since we're done using it
        os.remove(current_img)
        pred.save(os.path.join(IMG_DIR, new_name))
        return parse_contents(os.path.join(IMG_DIR, new_name))


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


@app.callback(
    Output("download", "data"),
    [Input("download-btn", "n_clicks"), State("upload-image", "filename")],
    prevent_initial_call=True)
def download_file(n, uploaded_filename):
    """Save uploaded files and regenerate the file list."""
    if n > 0:
        file_in_question = 'new_{}'.format(uploaded_filename[0])
        # print(file_in_question)
        return send_file('uploaded_img/{}'.format(file_in_question))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    # heroku sets its own port environment variable, so we need to run the server on
    # that port when on the server and otherwise 8000
    app.run_server(debug=False, host='0.0.0.0', port=port)
