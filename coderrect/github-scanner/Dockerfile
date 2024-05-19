FROM openjdk:11.0.9.1-jre-slim

WORKDIR /opt/app/

COPY target/*.jar app.jar

CMD java -Dspring.resources.static-locations=file:/www/,classpath:/static \
    -jar app.jar \
    --spring.profiles.active=default