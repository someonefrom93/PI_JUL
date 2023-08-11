# Well Come to MLOps Individual Project by Juan D

You will find here one of the thousand approaches to deliver a made up bussiness request.

What is that request? 

* Instintiate an API in order to consume data in this form.

* The next end points does not have any text format handling so that you have to pass perfect string writting to them.

* Count the number of movies by language. This end point accept languages iso code, for example: "de" is for german, "fr" for french, and so on...
Here you will find those iso languages code for trying out this end point [iso language codes](https://en.wikipedia.org/wiki/ISO_639-1_codes)

* Run Time and Release Year by movie title

* The collections of that movie title if it exists. For example: Try out to search "The Matrix" or any title of harry potter's movie

* Number of movies produced in a country. This is a good one because it retrieves the number og movies produced in the provided country. For example in Mexico a few movies has been produced in thatr lands eventhough they are not created by Mexican producers.

* The company success retrieves the information such as the name itself, total revenue, and number of movies produced. Try  inputting "Pixar Animation Studios" out.. or any other movie producer movies like Warner.

* Director Success. this one will give you all movies produced by the given director Name, for example Alfonso Cuaron.. I did not know this director has been a one of the directors of harry potter's saga. Even I did not know that those movies have multiple directors.

 * Finally you will see the recomender. This is a simple lightwieght recommendation model. It only uses 2000 records out of the almos 45k records of the original csv file. This is because if the dataframe may include the hole sataset, a very large array of cosine similatiries would take 14 gb of ram. Tha's wild. 


# My approach

The architecture that I follow is completelly towards functionality and effitiency. For the API you will find the etl.ipynb file with all functionalities taken for getting simple csv file into an end points. How was that get done?

![ETl flow for end points at Render](images/etl_flow.JPG)