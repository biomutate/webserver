// Get the modal elements
var modal1 = document.getElementById("myModal1");
var modal2 = document.getElementById("myModal2");
var modal3 = document.getElementById("myModal3");
var modal4 = document.getElementById("myModal4");
var modal5 = document.getElementById("myModal5");
var modal6 = document.getElementById("myModal6");
var modal7 = document.getElementById("myModal7");
var modal8 = document.getElementById("myModal8");


// Get the <span> elements that close the modals
var closeBtn1 = document.getElementsByClassName("close1")[0]; // Updated selector
var closeBtn2 = document.getElementsByClassName("close2")[0]; // Updated selector
var closeBtn3 = document.getElementsByClassName("close3")[0]; // Updated selector
var closeBtn4 = document.getElementsByClassName("close4")[0]; // Updated selector
var closeBtn5 = document.getElementsByClassName("close5")[0]; // Updated selector
var closeBtn6 = document.getElementsByClassName("close6")[0]; // Updated selector
var closeBtn7 = document.getElementsByClassName("close7")[0]; // Updated selector
var closeBtn8 = document.getElementsByClassName("close8")[0]; // Updated selector


// Function to open the modal for step 1
function openModal1() {
  modal1.style.display = "block";
}

// Function to close the modal for step 1
function closeModal1() {
  modal1.style.display = "none";
}

// Function to open the modal for step 2
function openModal2() {
  modal2.style.display = "block";
}

// Function to close the modal for step 2
function closeModal2() {
  modal2.style.display = "none";
}

// Function to open the modal for step 3
function openModal3() {
    modal3.style.display = "block";
}

// Function to close the modal for step 3
function closeModal3() {
    modal3.style.display = "None";
}

// Function to open the modal for step 3
function openModal4() {
    modal4.style.display = "block";
}

// Function to close the modal for step 3
function closeModal4() {
    modal4.style.display = "None";
}

// Function to open the modal for step 3
function openModal5() {
    modal5.style.display = "block";
}

// Function to close the modal for step 3
function closeModal5() {
    modal5.style.display = "None";
}

// Function to open the modal for step 3
function openModal6() {
    modal6.style.display = "block";
}

// Function to close the modal for step 3
function closeModal6() {
    modal6.style.display = "None";
}

// Function to open the modal for step 3
function openModal7() {
    modal7.style.display = "block";
}

// Function to close the modal for step 3
function closeModal7() {
    modal7.style.display = "None";
}

// Function to open the modal for step 3
function openModal8() {
    modal8.style.display = "block";
}

// Function to close the modal for step 3
function closeModal8() {
    modal8.style.display = "None";
}

// Event listener for the close button of modal 1
closeBtn1.addEventListener("click", closeModal1);

// Event listener for the close button of modal 2
closeBtn2.addEventListener("click", closeModal2);

// Event listener for the close button of modal3
closeBtn3.addEventListener("click", closeModal3);

// Event listener for the close button of modal 4
closeBtn4.addEventListener("click", closeModal4);

// Event listener for the close button of modal 5
closeBtn5.addEventListener("click", closeModal5);

// Event listener for the close button of modal 6
closeBtn6.addEventListener("click", closeModal6);

// Event listener for the close button of modal 7
closeBtn7.addEventListener("click", closeModal7);

// Event listener for the close button of modal 8
closeBtn8.addEventListener("click", closeModal8);

// Event listener to close the modals when clicking outside the modal content
window.addEventListener("click", function (event) {
  if (event.target == modal1) {
    closeModal1();
  } else if (event.target == modal2) {
    closeModal2();
  } else if (event.target == modal3) {
    closeModal3();
  } else if (event.target == modal4) {
    closeModal4();
  } else if (event.target == modal5) {
    closeModal5();
  } else if (event.target == modal6) {
    closeModal6();
  } else if (event.target == modal7) {
    closeModal7();
  } else if (event.target == modal8) {
    closeModal8();
  }
});
