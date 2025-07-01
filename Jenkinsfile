pipeline {
  agent any

  environment {
    DOCKERIMAGE = "pipeline-hello-world"
    HOME = "."
  }

  stages {
    stage('Limpiar Contenedores') {
      steps {
        script {
          sh '''
           CONTAINERS=$(docker ps -q)
          if [ ! -z "$CONTAINERS" ]; then
              docker stop $CONTAINERS
          else
              echo "No hay contenedores en ejecución."
          fi
          '''
        }
      }
    }

    stage('Build') {
      steps {
        script {
          docker.build(DOCKERIMAGE)
        }
      }
    }

    stage('Test') {
      steps {
        script {
          docker.image(DOCKERIMAGE).inside {
            sh 'python -m unittest discover -s tests -p "*.py"'
          }
        }
      }
    }

    stage('Deploy') {
      steps {
        script {
          withCredentials([
            string(credentialsId: 'PYTHONUNBUFFERED', variable: 'PYTHONUNBUFFERED'),
            string(credentialsId: 'HOST_PORT_BACKEND', variable: 'HOST_PORT_BACKEND'),
            string(credentialsId: 'PORT', variable: 'PORT'),
            string(credentialsId: 'HOST', variable: 'HOST'),
            string(credentialsId: 'DBURL', variable: 'DBURL'),
            string(credentialsId: 'DBNAME', variable: 'DBNAME'),
            string(credentialsId: 'DBUSER', variable: 'DBUSER'),
            string(credentialsId: 'DBPASSW', variable: 'DBPASSW'),
            string(credentialsId: 'CORS_ORIGIN', variable: 'CORS_ORIGIN')
          ]) {
            sh '''
            echo "Creando archivo .env desde Jenkins"
            cat <<EOF > .env
PYTHONUNBUFFERED=$PYTHONUNBUFFERED
HOST_PORT_BACKEND=$HOST_PORT_BACKEND
PORT=$PORT
HOST=$HOST
DBURL=$DBURL
DBNAME=turistdata
DBUSER=$DBUSER
DBPASSW=$DBPASSW
CORS_ORIGIN=$CORS_ORIGIN
EOF

            docker-compose --env-file .env -f docker-compose.yml up -d --build
            '''
          }
        }
      }
    }
  }
}