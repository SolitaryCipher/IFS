
module.exports = function (app) {

// Paths and requirements
var appPath = __dirname + "/../app/";
var path = require('path');
var componentsPath = path.join( appPath + "/components");
var passport = require('passport');

// Dev team Controllers
require(componentsPath + "/Login/loginRoutes")(app, passport);

//Tool Page and information
require(componentsPath + "/Tool/toolRoutes") (app);

// Preferences page Routes
require(componentsPath + "/Preferences/preferencesRoutes")(app);


}