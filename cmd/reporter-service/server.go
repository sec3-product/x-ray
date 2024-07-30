package crserver

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"path/filepath"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// FIXME: add more restrictions
// plus the permissions of system files
func isSafePath(p string) bool {
	if filepath.Ext(p) != ".rs" {
		return false
	}
	return true
}

func StartServer(host string, port int) {
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()
	router.Use(cors.Default())

	api := router.Group("/api")
	v1 := api.Group("/v1")
	files := v1.Group("/files")
	files.GET("/pull", func(c *gin.Context) {
		fp, err := url.QueryUnescape(c.Query("filepath"))
		if err != nil {
			fmt.Println("Failed to unescape the file path, error=", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err})
			return
		}
		bytes, err := ioutil.ReadFile(fp)
		if err != nil {
			fmt.Println("Failed to load file from path, error=", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": err})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"code": string(bytes),
		})
	})
	addr := fmt.Sprintf("%s:%d", host, port)
	router.Run(addr)
}
