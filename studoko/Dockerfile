FROM gradle:jdk17-jammy

COPY ../ /app/

RUN rm -r /app/studoko-frontend

WORKDIR /app/

RUN gradle clean build \
    -x shadowJar \
    -x shadowDistTar \
    -x shadowDistZip \
    -x loadKtlintReporters \
    -x runKtlintCheckOverTestSourceSet \
    -x runKtlintCheckOverKotlinScripts \
    -x ktlintTestSourceSetCheck \
    -x ktlintKotlinScriptCheck \
    -x runKtlintCheckOverMainSourceSet \
    -x ktlintMainSourceSetCheck \
    -x test \
    --no-daemon

EXPOSE 8080

ENTRYPOINT ["gradle", ":studoko-backend:run", "--no-daemon"]
