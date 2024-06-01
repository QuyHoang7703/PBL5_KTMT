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
const baseUrl = 'http://10.10.58.77:5000';
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
        const data = await response.json();

        if (response.ok) {
            // Xử lý đăng nhập thành công, ví dụ: chuyển hướng đến trang khác
            window.location.href = 'index';
        } else {
            // Hiển thị thông báo lỗi
            errorMessage.textContent = data.message || 'Đăng nhập thất bại. Vui lòng thử lại.';
        }
    } catch (error) {
        errorMessage.textContent = 'Có lỗi xảy ra. Vui lòng thử lại.';
    }
});