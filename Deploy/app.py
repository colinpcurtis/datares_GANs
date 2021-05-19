import base64
import os, sys
import datetime
from urllib.parse import quote as urlquote
import re
from flask import Flask, send_from_directory
import dash
from PIL import Image
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

# TODO: need to delete uploaded images after some time frame or we'll run out of space

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=[dbc.themes.MINTY])
app.title = 'Make-A-Monet'

# Frontend Components


# Nav Bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                # dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Home", href="/"),
                dbc.DropdownMenuItem("GitHub", href="https://github.com/colinpcurtis/datares_GANs"),
                dbc.DropdownMenuItem("Model Architecture", href="/page-2"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="CycleGAN Interactive Image Transformation",
    brand_href="#",
    color="primary",
    dark=True,
    style={'padding-left': '3.1%'}
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
        style={"width": "25rem", "height": "25rem"}),
    dcc.Upload(id="upload-image",
               children=dbc.Button("Upload", color="primary", size='lg'),
               multiple=True)
])

# Processbox
# processbox = dbc.Card(
#     [
#
#         dbc.CardBody(
#             [
#                 dbc.Row(
#                 [
#                     dbc.Col( dbc.Button("Process", id="process", color="primary"), width="auto"),
#                 ],align="center"
#             )
#             ]
#         )
#     ],
#     #style={"width": "12rem","height":"12rem"}
# )


processbox = dbc.Row(
    [
        dbc.Col(dbc.Button("Process", id="process", color="primary", size='lg'), width="auto"),
    ], align="center"
)

# outputbox
# outputbox = dbc.Card(
#     [
#         html.Div(id='output-image-upload'),
#     ],
#     style={"width": "18rem","height":"18rem"}
# )
outputbox = html.Div([
    html.H4("Transformed Image", style={'font-weight': 'bold'}),
    dbc.Card(
        [
            dbc.CardBody(
                [
                    dcc.Loading(id="loading-1",
                                children=[html.Div(id='output-image-upload')],
                                type="default")
                    # html.Div(id='output-image-upload')
                ]
            ),
        ],
        style={"width": "25rem", "height": "25rem"}),
    html.Div([dbc.Button("Download", id="download-btn", color="primary", size='lg'), Download(id="download")])
])

my_content = html.Div(id="page-content", children=[])
# app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    my_content
    # html.H2('Interactive Image Translation',
    #         style={'font-weight': 'bold', 'padding-left': '10%', 'font-size': '120%'}),
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
                        style={'padding-left': '15%', 'padding-right': '15%', 'padding-top': '5%'}, align="center"
                    ),
                ]
            )
        ]
    # # If the user tries to reach a different page, return a 404 message
    # return dbc.Jumbotron(
    #     [
    #         html.H1("404: Not found", className="text-danger"),
    #         html.Hr(),
    #         html.P(f"The pathname {pathname} was not recognised..."),
    #     ]
    # )


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
    return html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                 style={'width': '100%', 'height': "22.5rem"})
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
def update_output(n, uploaded_filename):
    """Save uploaded files and regenerate the file list."""
    if n > 0:
        file_in_question = 'new_{}'.format(uploaded_filename[0])
        print(file_in_question)
        return send_file('uploaded_img/{}'.format(file_in_question))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    # heroku sets its own port environment variable, so we need to run the server on
    # that port when on the server and otherwise 8888
    app.run_server(debug=False, host='0.0.0.0', port=port)
