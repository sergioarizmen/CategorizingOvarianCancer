from flask import Blueprint, request, Response
from src import classes, scripts

routes_blueprint = Blueprint('routes', __name__)

@routes_blueprint.route('/ping', methods=['GET'])
def ping():
    return 'Pong from ovarian cancer API'

@routes_blueprint.route('/classify-image', methods=['POST'])
def classifyImage():
    try:
        predictions = scripts.classifyImage(imageStream=request.files['img'])
        return Response(response=predictions, status=200, mimetype="application/json")
    except Exception as error:
        print(error)

    return Response(response="Invalid image, try with other image or contact support", status=500, mimetype="application/json")

# @routes_blueprint.route('/save-image', methods=['POST'])
# def saveImage():
#     print(request.files['img'].filename)
#     try:
#         img = classes.Image(image=request.files['img'])
#         print(img._imagePath)
#         # img.imageScale()
#         img.safeToImages()
#     except Exception as error:
#         print(error)

#     return 'Image saved to files'