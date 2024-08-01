import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import static groovyx.net.http.HttpBuilder.configure

def call(String pageId, String spaceKey, String title, String releaseNotes, String confluenceBaseUrl, String username, String apiToken) {
    def apiUrl = "${confluenceBaseUrl}/rest/api/content/${pageId}"

    // Fetch existing page content
    def existingPage = configure {
        request.uri = apiUrl
        request.headers['Authorization'] = "Basic ${"${username}:${apiToken}".bytes.encodeBase64().toString()}"
        request.headers['Content-Type'] = 'application/json'
    }.get()

    def version = existingPage.version.number + 1

    // Construct new content
    def newContent = """
    <h2>${title}</h2>
    <p>${releaseNotes}</p>
    """

    // Prepare the payload
    def payload = [
        id      : pageId,
        type    : 'page',
        title   : title,
        space   : [key: spaceKey],
        body    : [storage: [value: newContent, representation: 'storage']],
        version : [number: version]
    ]

    // Update the page
    configure {
        request.uri = apiUrl
        request.headers['Authorization'] = "Basic ${"${username}:${apiToken}".bytes.encodeBase64().toString()}"
        request.headers['Content-Type'] = 'application/json'
        request.body = JsonOutput.toJson(payload)
    }.put()
}
