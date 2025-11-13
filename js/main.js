// Ví dụ: cuộn mềm, chuyển menu active khi scroll, v.v.
// Chẳng hạn:

document.addEventListener("DOMContentLoaded", function() {
  const navLinks = document.querySelectorAll(".main-nav ul li a");
  navLinks.forEach(link => {
    link.addEventListener("click", function(e) {
      e.preventDefault();
      const target = this.getAttribute("href");
      document.querySelector(target)?.scrollIntoView({
        behavior: "smooth"
      });
    });
  });
});
