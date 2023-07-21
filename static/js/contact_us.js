function submitForm(event) {
    event.preventDefault();

    //Get Form Data
    var name = document.getElementById('name').value;
    var email = document.getElementById('email').value;
    var message = document.getElementById('message').value;

    //Send form data using AJAX
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/submit', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function () {
        if(xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            if(response.success) {
                //Display success notification
                var notification = document.createElement('div');
                notification.classList.add('notification');
                notification.textContent = 'Thank you for the submission';
                document.body.appendChild(notification);

                //Clear form field
                document.getElementById('name').value = '';
                document.getElementById('email').value = '';
                document.getElementById('message').value = '';
            } else {
                alert('Oops!! Something went wrong!!');
            }
        } else {
            alert('Oops!! Something went wrong!!')
        }
    };

    var data = JSON.stringify({ name: name, email:email, message: message});
    xhr.send(data);
}