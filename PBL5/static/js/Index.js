let menuToggle = document.querySelector('.menuToggle');
let sidebar = document.querySelector('.sidebar');
let iframeWrapper = document.querySelector('.iframe-wrapper');

menuToggle.onclick = function() {
    menuToggle.classList.toggle('active');
    sidebar.classList.toggle('active');

    if (sidebar.classList.contains('active')) {
        iframeWrapper.style.width = 'calc(100% - 300px)';
        iframeWrapper.style.marginLeft = '300px';
    } else {
        iframeWrapper.style.width = 'calc(100% - 80px)';
        iframeWrapper.style.marginLeft = '80px';
    }
}

let Menulist = document.querySelectorAll('.Menulist li');

function activelink() {
    Menulist.forEach((item) => item.classList.remove('active'));
    this.classList.add('active');
    let href = this.querySelector('a').getAttribute('href');
    iframe.src = href;
}

Menulist.forEach((item) => item.addEventListener('click', activelink));

document.getElementById("Logout").addEventListener("click", function() {
    window.location.href = '/';
});