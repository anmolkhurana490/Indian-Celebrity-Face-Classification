from flask import Flask, render_template, request
#from celebrity_prediction import preprocess_predict_image
import json

app=Flask(__name__)

with open("Indian celebrities.json", 'r') as file:
	celebrities=json.loads(file.read())

#results=pd.read_csv("results.csv").to_dict(orient="records")

celebrities_data=[]
for celebrity in celebrities:
	data={
		"name": celebrity,
		"img_location": f"Images/cropped faces/{celebrity}/0.jpg"
	}
	celebrities_data.append(data)
	

@app.route('/', methods=["GET", "POST"])
def home():
	file=request.files.get("image-upload")
	if request.method=='POST':
		results=preprocess_predict_image(file)
		if type(results)==str:
			return render_template("app.html", celebrities=celebrities_data, error=results)
		else:
			return render_template("app.html", celebrities=celebrities_data, classified_faces=results.to_dict(orient="records"))
	else:
		return render_template("app.html", celebrities=celebrities_data)

if __name__=="__main__":
	app.run(debug=True)