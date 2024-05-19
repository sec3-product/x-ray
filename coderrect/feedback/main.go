package main


import (
//	"encoding/json"

	//	"encoding/json"
	"fmt"
	"log"
//	"net/http"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-lambda-go/lambda"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ses"
	"github.com/aws/aws-sdk-go/aws/awserr"
)


const (
	// This address must be verified with Amazon SES.
	Sender = "jie@coderrect.com"

	// Replace recipient@example.com with a "To" address. If your account
	// is still in the sandbox, this address must be verified.
	Recipient = "jeff@coderrect.com"

	// Specify a configuration set. To use a configuration
	// set, comment the next line and line 92.
	//ConfigurationSet = "ConfigSet"

	// The subject line for the email.
	Subject = "A new feedback from our customer"

	// The character encoding for the email.
	CharSet = "UTF-8"
)


func buildBody(message string) string {
	a := `
{
	"message": “%s"
}
`
	return fmt.Sprintf(a, message)
}


func HandleRequest(req events.APIGatewayProxyRequest) (events.APIGatewayProxyResponse, error) {
	email := req.QueryStringParameters["email"]
	comment := req.QueryStringParameters["comment"]

	log.Printf("Gets a new feedback. request=%+v, email=%s, comment=%s", req, email, comment)

	// Create a new session in the us-west-2 region.
	// Replace us-west-2 with the AWS Region you're using for Amazon SES.
	sess, err := session.NewSession(&aws.Config{
		Region:aws.String("us-west-2")},
	)
	if err != nil {
		log.Printf("Unable to new an aws session. error=%v", err)
		return events.APIGatewayProxyResponse {
			StatusCode: 500,
			Body: buildBody(err.Error()),
		}, err
	}

	// Create an SES session.
	svc := ses.New(sess)

	// Assemble the email.
	emailBody := fmt.Sprintf("We get a new feedback from %s\n\n%s\n", email, comment)
	log.Printf("Sending an email. body=%s", emailBody)

	input := &ses.SendEmailInput{
		Destination: &ses.Destination{
			CcAddresses: []*string{
			},
			ToAddresses: []*string{
				aws.String(Recipient),
			},
		},
		Message: &ses.Message{
			Body: &ses.Body{
				Text: &ses.Content{
					Charset: aws.String(CharSet),
					Data:    aws.String(emailBody),
				},
			},
			Subject: &ses.Content{
				Charset: aws.String(CharSet),
				Data:    aws.String(Subject),
			},
		},
		Source: aws.String(Sender),
		// Uncomment to use a configuration set
		//ConfigurationSetName: aws.String(ConfigurationSet),
	}

	// Attempt to send the email.
	_, err = svc.SendEmail(input)
	if err != nil {
		log.Printf("Failed to send an email. error=%v", err)

		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case ses.ErrCodeMessageRejected:
				fmt.Println(ses.ErrCodeMessageRejected, aerr.Error())
			case ses.ErrCodeMailFromDomainNotVerifiedException:
				fmt.Println(ses.ErrCodeMailFromDomainNotVerifiedException, aerr.Error())
			case ses.ErrCodeConfigurationSetDoesNotExistException:
				fmt.Println(ses.ErrCodeConfigurationSetDoesNotExistException, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
		} else {
			// Print the error, cast err to awserr.Error to get the Code and
			// Message from an error.
			fmt.Println(err.Error())
		}

		return events.APIGatewayProxyResponse{
			StatusCode: 200,
			Body: buildBody(err.Error()),
		}, err
	}

	return events.APIGatewayProxyResponse{
		StatusCode:        200,
		Body:              buildBody("OK"),
	}, nil
}


func main() {
	lambda.Start(HandleRequest)
}


