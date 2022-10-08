from sanic import Sanic, response
from sanic.response import json
from warmup import load_model
from run import encode_image, encode_text , get_similarity
import requests
from io import BytesIO
import base64
import json

# do the warmup step globally, to have a reuseable model instance
model = load_model()

app = Sanic("my_app")


@app.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    print('Calling health check')
    return response.json({"state": "healthy"})

@app.route('/', methods=["POST"]) # Do not edit - POST requests to "/" are a required interface
def inference(request):
    print('Calling inference')
    try:
        model_parameters = json.loads(request.json)
    except:
        model_parameters = request.json
    
    print('Model Parameters')
    model_params_json = json.dumps(model_parameters)
    print(model_params_json)
    
    image_byte_string = model_parameters.get('imageByteString', None)
    image_url = model_parameters.get('imageURL', None)
    text = model_parameters.get('text', None)
    texts = model_parameters.get('texts', None)

    if image_byte_string == None and image_url == None:
        return json({'message': "No image provided"})

    print('Has Text Images')
    if text == None and texts ==  None:
        return json({'message': "No text provided"})
    
    print('Has Text Inputs')
    image_bytes = None

    if image_url != None:
        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)

    if image_byte_string != None:
        image_encoded = image_byte_string.encode('utf-8')
        image_bytes = BytesIO(base64.b64decode(image_encoded))

    # preencode the image
    image_encoding = encode_image(image_bytes,model)

    response = {}

    if texts != None:
        sims = []
        for t in texts:
            text_encoding = encode_text(t,model)
            sim = get_similarity(text_encoding, image_encoding)
            sims.append(sim)
            response["similarities"] = sims

    if text != None:
        text_encoding = encode_text(text,model)
        sim = get_similarity(text_encoding, image_encoding)
        response['similarity'] = sim

    return json(response) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)
