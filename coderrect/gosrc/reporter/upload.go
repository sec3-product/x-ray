package reporter

import (
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/coderrect-inc/coderrect/util/platform"
	"github.com/google/uuid"
)

// NOTE: This is only used for our own Jenkins task.
//  The assumption is that our Jenkins server and upload server are hosted in the same machine
const localHost = "http://127.0.0.1:8080"
const defaultHost = "http://cloud.coderrect.com"
const devHost = "http://cloud.coderrect.com:8081"
const uploadAPI = "/api/v1/report/upload"

// publish this report to coderrect remote server
func PublishReport(path, host string) string {
	id := getUUID()
	tarPath := zipReport(id, path)
	if host == "DEFAULT" {
		host = defaultHost
	} else if host == "LOCAL" {
		host = localHost
	}
	repo := collectRepoURL()
	url := uploadReport(id, tarPath, repo, host)
	fmt.Println("")
	fmt.Println("Report has been uploaded to Coderrect server!")
	fmt.Println("")
	fmt.Println("You can view it through: " + url + "/index.html")
	fmt.Println("")
	return url + "/index.html"
}

// `path` should be the .coderrect folder
// the return value is the path of the generated tarball
func zipReport(name, path string) string {
	tarName := name + ".tar.gz"
	// cmdline := fmt.Sprintf("tar czf %s -C \"%s\" .", tarName, path)
	cmdline := "tar"
	cmd := platform.Command(cmdline, "czf", tarName, "-C", path, ".")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		os.Stderr.WriteString("Fail to create a tarball for the HTML report.")
		panic(err)
	}

	return filepath.Join("./", tarName)
}

func collectRepoURL() string {
	cmd := fmt.Sprintf("git config --get remote.origin.url")
	out, err := platform.Command(cmd).Output()
	if err != nil {
		return "Unknown repository (not a git project)"
	}
	url := string(out)
	url = strings.TrimSpace(url)
	return url
}

func uploadReport(id, path, repo, host string) string {
	req, err := newUploadRequest(id, path, repo, host)

	client := http.Client{}
	res, err := client.Do(req)
	if err != nil {
		panic(err)
	}

	if res.StatusCode != http.StatusOK {
		fmt.Println(fmt.Sprintf("[ERROR] http status code: %d", res.StatusCode))
	}
	defer res.Body.Close()
	// body, err := ioutil.ReadAll(res.Body)
	// fmt.Println("---:")
	// fmt.Println(string(body))
	var url string
	if host == localHost {
		url = defaultHost + "/" + id
	} else {
		url = host + "/" + id
	}

	return url
}

func newUploadRequest(id, path, repo, host string) (*http.Request, error) {
	body, contentType, err := newUploadBody(id, path, repo)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, host+uploadAPI, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)

	return req, nil
}

// prepare the POST request body
func newUploadBody(id, path, repo string) (io.Reader, string, error) {
	bodyReader, bodyWriter := io.Pipe()
	uploadWriter := multipart.NewWriter(bodyWriter)

	file := mustOpen(path)
	// NOTE: this goroutine does not require explicit wait,
	// this is because there's an internal channel inside `bodyReader` and `bodyWriter`
	// The `bodyReader` will block until a writer arrive.
	// Similarly, the `bodyWriter` will block until a reader until one or more readers consumes all the data.
	// Therefore, the implementation of io.Pipe guarantees this goroutine finishes before the http request is sent
	go func() {
		defer bodyWriter.Close()
		defer file.Close()

		err := uploadWriter.WriteField("reportId", id)
		if err != nil {
			bodyWriter.CloseWithError(err)
			return
		}

		err = uploadWriter.WriteField("repoUrl", repo)
		if err != nil {
			bodyWriter.CloseWithError(err)
			return
		}

		part, err := uploadWriter.CreateFormFile("compressedFile", file.Name())
		if err != nil {
			bodyWriter.CloseWithError(err)
			return
		}

		_, err = io.Copy(part, file)
		if err != nil {
			bodyWriter.CloseWithError(err)
			return
		}

		err = uploadWriter.Close()
		if err != nil {
			bodyWriter.CloseWithError(err)
			return
		}
	}()

	return bodyReader, uploadWriter.FormDataContentType(), nil
}

func mustOpen(f string) *os.File {
	r, err := os.Open(f)
	if err != nil {
		panic(err)
	}
	return r
}

// Generate UUID and remove hyphens
func getUUID() string {
	uuidWithHyphen := uuid.New()
	return strings.Replace(uuidWithHyphen.String(), "-", "", -1)
}
