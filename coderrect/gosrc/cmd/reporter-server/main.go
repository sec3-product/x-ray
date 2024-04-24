package main

import (
	"flag"

	crserver "github.com/coderrect-inc/coderrect/cmd/reporter-service"
)

func main() {
	host := flag.String("h", "localhost", "addr for the server")
	port := flag.Int("p", 8887, "The port number that this server listens to")
	flag.Parse()
	crserver.StartServer(*host, *port)
}
