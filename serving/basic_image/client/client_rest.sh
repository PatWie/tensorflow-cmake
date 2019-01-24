B64="$(base64 donkey_small.png)"
ENDPOINT=http://localhost:8501/v1/models/simple_img_model:predict
# REQUEST="{\"instances\": [{\"image\": { \"b64\": \"$B64\" }}]}"
REQUEST="{\"inputs\": [{ \"b64\": \"$B64\" }]}"

echo $REQUEST
curl -X POST $ENDPOINT -d '${REQUEST}'