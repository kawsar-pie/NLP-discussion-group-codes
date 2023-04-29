from n_gram import NGramLanguageModel

bigram_model = NGramLanguageModel(n=2)
# corpus = ["My name is Kawsar", "I am a very good boy",
#           "but my mother says I am a very bad boy", "What can I do now?"]
train_corpus = [
    "Python finds extensive use in scientific computing.",
    "Raspberry Pi sees frequent use in embedded systems.",
    "Git is a common choice for version control in software development.",
    "JavaScript is widely used for client-side scripting.",
    "Django is a popular framework for building web applications.",
    "Cloud computing is often chosen for scalable and flexible computing resources.",
    "SQL is a widely used language for managing and querying databases.",
    "Node.js is frequently chosen for building scalable network applications.",
    "Flask is a lightweight web framework often used for building web applications.",
    "HTML is the foundation for creating web pages.",
    "CSS is commonly used for styling web pages.",
    "React is a popular choice for building interactive user interfaces.",
    "Bootstrap is a widely used framework for building responsive web designs.",
    "Angular is often used for building single-page web applications.",
    "MongoDB is a common choice for NoSQL database management.",
    "PostgreSQL is frequently used for relational database management.",
    "Amazon Web Services (AWS) is a widely used platform for cloud computing.",
    "Microsoft Azure is a popular choice for cloud computing and storage.",
    "Google Cloud Platform is often chosen for cloud computing and machine learning.",
    "Artificial Intelligence (AI) is frequently used for machine learning and data analysis.",
    "Machine Learning is often utilized for predictive analytics and natural language processing.",
    "Data Science is widely used for extracting insights from data.",
    "Big Data is commonly analyzed for insights and trends.",
    "Hadoop is frequently chosen for distributed data storage and processing.",
    "Apache Spark is often used for distributed data processing.",
    "Java is a popular choice for enterprise software development.",
    "C++ is commonly used for system programming and game development.",
    "C# is frequently chosen for Microsoft .NET development.",
    "Visual Basic is often used for developing Windows applications.",
    "Ruby on Rails is a popular framework for web application development.",
    "Swift is a popular choice for iOS and macOS application development.",
    "Kotlin is frequently used for Android application development.",
    "Objective-C is often used for iOS and macOS application development.",
    "Unity is a popular choice for game development.",
    "Unreal Engine is often used for creating AAA games.",
    "OpenGL is frequently chosen for graphics rendering and visualization.",
    "DirectX is a popular choice for Windows game development.",
    "Virtual Reality (VR) is often used for gaming and simulation.",
    "Augmented Reality (AR) is commonly used for mobile app development.",
    "Flutter is a popular choice for mobile app development.",
    "Ionic is often used for cross-platform mobile app development.",
    "PhoneGap is a widely used platform for hybrid mobile app development.",
    "Xamarin is frequently chosen for cross-platform mobile app development.",
    "TensorFlow is a popular framework for machine learning and deep learning.",
    "PyTorch is often used for machine learning and neural network development.",
    "Scikit-learn is commonly used for machine learning and data analysis.",
    "NLTK is a popular choice for natural language processing.",
    "Keras is frequently used for deep learning and neural network development.",
    "Pygame is often used for game development with Python.",
    "Pygame Zero is a simple game development framework for Python.",
    "Pygame GUI is often used for creating graphical user interfaces with Python.",
    "PyQt is a popular choice for creating desktop applications with Python.",
    "Tkinter is often used for creating simple desktop applications with Python.",
    "Pandas is widely used for data analysis and manipulation with Python.",
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

test_corpus = "C++ is a high-level"
bigram_model.train(train_corpus)
# returns a log probability
score = bigram_model.score(test_corpus)
print(score)

generated_text = bigram_model.predict(test_corpus, max_length=100)
print(generated_text)

perplexity = bigram_model.perplexity(["Git is a common choice for version control in software development."])
print(perplexity)