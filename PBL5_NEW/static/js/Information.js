// const baseUrl = 'http://127.0.0.1:5000';

async function getSessionData() {
    try {
        const response = await fetch(`${baseUrl}/get_session`);
        if (response.ok) {
            const sessionData = await response.json();
            return sessionData.id_account;
        } else {
            console.error('No session data found');
            return null;
        }
    } catch (error) {
        console.error('Error fetching session data:', error);
        return null;
    }
}

async function fetchAccountDetails(accountId) {
    try {
        const response = await fetch(`${baseUrl}/information-management/information/account/${accountId}`);
        if (response.ok) {
            const accountData = await response.json();
            return accountData;
        } else {
            console.error('Failed to fetch account details');
            return null;
        }
    } catch (error) {
        console.error('Error fetching account details:', error);
        return null;
    }
}

async function populateAccountDetails() {
    const accountId = await getSessionData();
    if (!accountId) {
        document.getElementById('reg-error-message').textContent = 'Không thể lấy thông tin tài khoản từ session.';
        return;
    }

    const accountDetails = await fetchAccountDetails(accountId);
    if (!accountDetails) {
        document.getElementById('reg-error-message').textContent = 'Không thể lấy thông tin tài khoản.';
        return;
    }
    document.getElementById('regName').value = accountDetails.name || '';
    document.getElementById('regPhone').value = accountDetails.phone || '';
    document.getElementById('gender').value = accountDetails.gender ? 'female' : 'male';
    document.getElementById('regCCCD').value = accountDetails.cccd || '';
    document.getElementById('regGmail').value = accountDetails.gmail || '';
}

document.addEventListener('DOMContentLoaded', populateAccountDetails);


function myLogPassword() {
    var a = document.getElementById("logPassword");
    var b = document.getElementById("eye");
    var c = document.getElementById("eye-slash");

    if (a.type === "password") {
        a.type = "text";
        b.style.opacity = "0";
        c.style.opacity = "1";
    } else {
        a.type = "password";
        b.style.opacity = "1";
        c.style.opacity = "0";
    }
}

function myRegPassword(){
    
    var d = document.getElementById("regPassword");
    var b = document.getElementById("eye-2");
    var c = document.getElementById("eye-slash-2");

    if(d.type === "password"){
       d.type = "text";
       b.style.opacity = "0";
       c.style.opacity = "1";
    }else{
       d.type = "password";
       b.style.opacity = "1";
       c.style.opacity = "0";
    }

}
document.getElementById("eye-slash-2-2").style.opacity = "0"
function myRegPasswordAgain(){
    
    var d = document.getElementById("regPasswordAgain");
    var b = document.getElementById("eye-2-2");
    var c = document.getElementById("eye-slash-2-2");

    if(d.type === "password"){
       d.type = "text";
       b.style.opacity = "0";
       c.style.opacity = "1";
    }else{
       d.type = "password";
       b.style.opacity = "1";
       c.style.opacity = "0";
    }

}

var x = document.getElementById('login');
var y = document.getElementById('register');
var z = document.getElementById('btn');

function login(){
    x.style.left = "27px";
    y.style.right = "-350px";
    z.style.left = "0px";
}
function register(){
    x.style.left = "-350px";
    y.style.right = "25px";
    z.style.left = "150px";
}

document.getElementById('continue-btn').addEventListener('click', async function(event) {
    event.preventDefault();  // Prevent form from submitting normally

    // Get session data to verify the user
    const accountId = await getSessionData();
    if (!accountId) {
        console.error('Session ID could not be retrieved');
        document.getElementById('reg-error-message').textContent = 'Session error. Please log in again.';
        return;
    }

    // Gather input data from the form
    const name = document.getElementById('regName').value;
    const phone = document.getElementById('regPhone').value;
    const gender = document.getElementById('gender').value === 'male' ? false : true; // Convert to boolean, assuming male is false, female is true
    const cccd = document.getElementById('regCCCD').value;
    const gmail = document.getElementById('regGmail').value;

    // Prepare the request options
    const requestOptions = {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            id_account: accountId,
            name,
            gender,
            phone,
            cccd,
            gmail
        })
    };

    // Send the request to the server
    try {
        const response = await fetch(`${baseUrl}/information-management/information/account/${accountId}`, requestOptions);
        const result = await response.json();
        if (response.ok) {
            alert('Thông tin đã được cập nhật thành công!');
        } else {
            document.getElementById('reg-error-message').textContent = result.message || 'Failed to update information.';
        }
    } catch (error) {
        console.error('Error updating information:', error);
        document.getElementById('reg-error-message').textContent = 'Network error. Please try again later.';
    }
});



document.getElementById('loginForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const oldPassword = document.getElementById('logPassword').value;
    const newPassword = document.getElementById('regPassword').value;
    const newPasswordAgain = document.getElementById('regPasswordAgain').value;
    const errorMessage = document.getElementById('error-message');

    if (newPassword !== newPasswordAgain) {
        errorMessage.textContent = 'Mật khẩu mới không khớp.';
        return;
    }

    try {
        const response = await fetch(`${baseUrl}/account-management/account/change-password`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword
            })
        });

        const data = await response.json();
        if (response.ok) {
            alert(data.message);
            // Optionally, redirect the user or clear the form
            document.getElementById('loginForm').reset();
        } else {
            errorMessage.textContent = data.message;
        }
    } catch (error) {
        errorMessage.textContent = 'Đã xảy ra lỗi khi thay đổi mật khẩu. Vui lòng thử lại.';
    }
});