// const baseUrl = "{{ base_url }}";
console.log(baseUrl);
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

document.getElementById('loginForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    const username = document.getElementById('logEmail').value;
    const password = document.getElementById('logPassword').value;
    const errorMessage = document.getElementById('error-message');
    try {
        const response = await fetch(`${baseUrl}/account-management/account/check`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        if (response.ok) {
            // Xử lý đăng nhập thành công, ví dụ: chuyển hướng đến trang khác
            const accountData = await response.json();
            const role = accountData.role;
            if(role == 1)
                window.location.href = 'index';
            else
                window.location.href = 'index_2';
        } else {
            // Hiển thị thông báo lỗi
            errorMessage.textContent = data.message || 'Đăng nhập thất bại. Vui lòng thử lại.';
        }
    } catch (error) {
        console.error('Error:', error);
        errorMessage.textContent = 'Có lỗi xảy ra. Vui lòng thử lại.';
    }
});

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

document.getElementById('register-btn').addEventListener('click', async function(event) {
    // Lấy giá trị các trường đăng ký
    var regUsername = document.getElementById('regUsername').value;
    var regPassword = document.getElementById('regPassword').value;
    var regConfirmPassword = document.getElementById('regPasswordAgain').value;
    var regName = document.getElementById('regName').value;
    var regGender = document.getElementById('gender').value;
    var regPhone = document.getElementById('regPhone').value;
    var regCCCD = document.getElementById('regCCCD').value;
    var regGmail = document.getElementById('regGmail').value;

    // Kiểm tra các trường bắt buộc
    if (!regUsername || !regPassword || !regConfirmPassword || !regName || !regGender || !regPhone || !regCCCD || !regGmail) {
        document.getElementById('reg-error-message').textContent = 'Vui lòng điền đầy đủ thông tin.';
        return; // Dừng lại nếu thiếu thông tin
    }

    try {
        // Thực hiện yêu cầu POST để tạo tài khoản và thông tin tài khoản
        const response = await fetch(`${baseUrl}/account-management/account`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: regUsername,
                password: regPassword,
                role: false // Giả sử mặc định tạo tài khoản là khách hàng
            })
        });

        if (response.ok) {
            // Nếu tạo tài khoản thành công, tiếp tục thêm thông tin cá nhân
            const accountData = await response.json();
            const accountId = accountData.id_account;

            const infoResponse = await fetch(`${baseUrl}/information-management/information`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    id_account: accountId,
                    name: regName,
                    gender: regGender === 'male' ? false : true,
                    phone: regPhone,
                    cccd: regCCCD,
                    gmail: regGmail
                })
            });

            if (infoResponse.ok) {
                // Nếu tạo thông tin cá nhân thành công, thông báo cho người dùng và thực hiện các hành động khác cần thiết
                alert('Tạo tài khoản thành công!');
                window.location.href = '';
                // Thực hiện chuyển hướng hoặc các hành động khác tại đây
            } else {
                // Nếu có lỗi xảy ra trong quá trình tạo thông tin cá nhân, hiển thị thông báo lỗi tương ứng
                const infoData = await infoResponse.json();
                document.getElementById('reg-error-message').textContent = infoData.message;
            }
        } else {
            // Nếu có lỗi xảy ra trong quá trình tạo tài khoản, hiển thị thông báo lỗi tương ứng
            const accountData = await response.json();
            document.getElementById('reg-error-message').textContent = accountData.message;
        }
    } catch (error) {
        // Nếu có lỗi trong quá trình thực hiện yêu cầu, hiển thị thông báo lỗi
        console.error('Lỗi:', error);
        document.getElementById('reg-error-message').textContent = 'Đã xảy ra lỗi khi tạo tài khoản';
    }
});


document.getElementById('continue-btn').addEventListener('click', async function(event) {
    // Lấy giá trị các trường đăng ký
    var regUsername = document.getElementById('regUsername').value;
    var regPassword = document.getElementById('regPassword').value;
    var regConfirmPassword = document.getElementById('regPasswordAgain').value;
    
    // Kiểm tra các trường bắt buộc
    if (!regUsername || !regPassword || !regPasswordAgain) {
        document.getElementById('reg-error-message').textContent = 'Vui lòng điền đầy đủ thông tin.';
        return; // Dừng lại nếu thiếu thông tin
    }

    try {
        const response = await fetch(`/account-management/account/check-username/${regUsername}`);
        const data = await response.json();

        if (data.exists) {
            document.getElementById('reg-error-message').textContent = 'Username already exists';
            return;
        } 
    } catch (error) {
        document.getElementById('reg-error-message').textContent = 'An error occurred while checking the username';
        console.error('Error:', error);
    }
    
    if (regPassword !== regConfirmPassword) {
        document.getElementById('reg-error-message').textContent = 'Mật khẩu không khớp. Vui lòng nhập lại.';
        return; // Dừng lại nếu mật khẩu không khớp
    }
    // Ẩn các trường đã đăng ký
    document.getElementById('regUsername').parentElement.style.display = 'none';
    document.getElementById('regPassword').parentElement.style.display = 'none';
    document.getElementById('regPasswordAgain').parentElement.style.display = 'none';

    // Hiển thị các trường ẩn
    document.getElementById('regName').parentElement.style.display = 'block';
    document.getElementById('regPhone').parentElement.style.display = 'block';
    document.getElementById('gender').parentElement.style.display = 'block';
    document.getElementById('regCCCD').parentElement.style.display = 'block';
    document.getElementById('regGmail').parentElement.style.display = 'block';

    // Ẩn nút "Tiếp tục"
    document.getElementById('continue-btn').style.display = 'none';
    
    // Hiển thị nút "Đăng kí"
    document.getElementById('register-btn').style.display = 'block';

    // Ẩn tiêu đề "Đăng kí"
    document.getElementById('register-header').style.display = 'none';
});