from n_gram import NGramLanguageModel

bigram_model = NGramLanguageModel(n=2)
# corpus = ["My name is Kawsar", "I am a very good boy",
#           "but my mother says I am a very bad boy", "What can I do now?"]
corpus = [
    "Python is a popular programming language used for web development.",
    "The Raspberry Pi is a popular device used for prototyping.",
    "Git is a version control system used by many developers.",
    "JavaScript is a scripting language used for front-end development.",
    "Django is a web framework built with Python.",
    "The Cloud is a network of remote servers used for storing, managing, and processing data.",
    "SQL is a programming language used for managing and manipulating data in databases.",
    "Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine.",
    "Flask is a micro web framework written in Python.",
    "HTML is a markup language used for creating web pages.",
    "AWS is a cloud computing platform provided by Amazon.",
    "jQuery is a JavaScript library used for simplifying HTML DOM traversal and manipulation.",
    "CSS is a stylesheet language used for describing the presentation of a document written in HTML.",
    "MySQL is a popular open-source relational database management system.",
    "Bootstrap is a front-end web development framework used for building responsive, mobile-first websites.",
    "PyCharm is a popular integrated development environment used for Python programming.",
    "Artificial Intelligence is a field of computer science focused on creating intelligent machines that can perform tasks that would typically require human intelligence.",
    "Apache is a widely used open-source web server software.",
    "Scikit-learn is a popular machine learning library for Python.",
    "MongoDB is a document-oriented NoSQL database used for handling large amounts of data.",
    "React is a JavaScript library used for building user interfaces.",
    "TensorFlow is an open-source software library used for dataflow and differentiable programming across a range of tasks.",
    "PHP is a popular server-side scripting language used for web development.",
    "PyTorch is an open-source machine learning library for Python.",
    "PostgreSQL is a powerful open-source object-relational database system.",
    "Angular is a popular front-end web application platform.",
    "Kubernetes is a popular open-source container orchestration platform.",
    "Visual Studio Code is a popular free code editor developed by Microsoft.",
    "Linux is a widely used open-source operating system.",
    "Numpy is a Python library used for working with arrays and mathematical functions.",
    "Apache Spark is a fast and general-purpose cluster computing system.",
    "Vue.js is a progressive JavaScript framework used for building user interfaces.",
    "Docker is a platform used for developing, shipping, and running applications in containers.",
    "GraphQL is a query language used for APIs.",
    "Amazon S3 is a simple storage service provided by Amazon Web Services.",
    "Anaconda is a popular open-source distribution of Python and R for scientific computing.",
    "C++ is a high-level programming language used for developing software.",
    "Elasticsearch is a distributed, open-source search and analytics engine.",
    "Firebase is a mobile and web application development platform developed by Google.",
    "Apache Kafka is an open-source distributed event streaming platform.",
    "Pandas is a Python library used for data manipulation and analysis.",
    "Jupyter Notebook is an open-source web application used for creating and sharing documents that contain live code, equations, visualizations, and narrative text.",
    "Apache Hadoop is an open-source framework used for storing and processing big data.",
    "Ruby on Rails is a popular web development framework built with the Ruby programming language.",
    "React Native is a popular framework used for building native mobile applications using JavaScript.",
    "Swift is a powerful and intuitive programming language used for developing apps for Apple platforms."]

bigram_model.train(corpus)
# returns a log probability
score = bigram_model.score("C++ is a")
print(score)

generated_text = bigram_model.predict("C++ is a", max_length=10)
print(generated_text)