Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "http://127.0.0.1:5000/pass", // Adjust the URL according to your server endpoint
        maxFiles: 1, // Maximum number of files allowed
        maxFilesize: 20, // Maximum file size in MB
        acceptedFiles: "application/pdf", // Accepted file types
        addRemoveLinks: true,
        dictDefaultMessage: "Drag and drop a file here or click to upload",
        autoProcessQueue: false
    });

    dz.on("addedfile", function() {
        console.log('File added:', dz.files[0].name);
        if (dz.files.length > 1) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function(file) {
        dz.removeFile(file);
    });

    dz.on("success", function(file, response) {
        console.log('Response from server:', response);
        if (response.scores) {
            displayScores(response.scores);
        } else if (response.error) {
            $("#error").show().html("Error: " + response.error);
        }
    });

    $("#submitBtn").on('click', function(e) {
        e.preventDefault();
        dz.processQueue();
    });
}

function displayScores(scores) {
    $("#resultHolder").empty(); // Clear any previous results

    // Create a container for the grid
    let gridContainer = $("<div></div>").addClass("grid-container");

    scores.forEach(function(score, index) {
        let gridItem = $("<div></div>").addClass("grid-item").text(score); // Create a div for each score
        gridContainer.append(gridItem); // Append the item to the grid container
    });

    $("#resultHolder").append(gridContainer); // Append the grid container to the result holder
    $("#resultHolder").show(); // Show the result holder
}
document.addEventListener("DOMContentLoaded", function() {
    console.log("Ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});
