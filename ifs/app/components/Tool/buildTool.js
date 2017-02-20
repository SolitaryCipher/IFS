
/* 
    Passing a JSON object that represents a tool with options
    
    ***
    Required members are (please update as this changes)
    progName-> name of the command to run this program.

    Optional:
    options: can be a list of arguments or none

    Return: A blank object if error
            Else: returns an object wtih combination of arguments are program name.
*/
var fs = require('fs');
var _ = require('lodash');

const progName = "progName";
const options = "options";
const runCmd = "runCmd";


// Searches the JSON tool list to find a specific tool and return it */
function getJsonTool( toolsJson, targetTool ){
    var tool = _.find(toolsJson, function(t) {
        return  t.progName == targetTool;
    });
    return tool;
}

// This function parses the form data from /tool into separate parameters for
// tool object. ***Note data remains in the received form but gets classified.
function parseFormSelection( formData ) {

    var isDone = false;
    var toolMarker = 'tool-';
    var targetTool = {};
    var progName ="";
    var option = {};
    var tool = undefined;

    var toolOptions = { 'files': formData['files'], 'tools':[] };
  
    _.forEach( formData, function(value, key ) {

        if( key == "submit") {
            if(tool) {
                toolOptions.tools.push( tool );
            }
            isDone = true;
        }

        if( !isDone ){

            if( _.startsWith(key, toolMarker) ) {

                if(tool) {
                    toolOptions.tools.push( tool );
                }

                progName = key.substr( toolMarker.length );
                tool = { 'progName': progName, 'options':[] };
            }
            else if(tool) {
                var r = {};
                r['name'] = key;
                r['value'] = value;
                tool.options.push( r );
            }
        }
    });
    return toolOptions;
}

// Reads JSON document and creates an array of objects with tool information and options.
// See tooList.json (these are tools avilable in the tool page)
function readToolFileList() {

    var supportedToolsFile = './tools/toolList.json';
    var result = fs.readFileSync( supportedToolsFile, 'utf-8');
    var jsonObj = JSON.parse(result);
    return jsonObj;
}

// This is the external call that uses the form data,
// user selected options to create jobs for the Queue
function createJobRequests( selectedOptions ) {

    var toolList = readToolFileList();
    var toolOptions = parseFormSelection( selectedOptions );
    
    var res = tempInsertOptions(toolList.tools, toolOptions);
    var jobReq =  buildJobs(res, selectedOptions.files, {prefixArg: false} );

    return jobReq;

}

// User options are already for the appropriate tool
// This handles converting checkbox and other input types
// into appropriate command line parameters
// ex) checkbox on means -option while off would be blank
function basicParse( toolListItem, userOptions ){

    _.forEach( toolListItem.options, function(option) {

        var userTargetTool = _.find( userOptions.options, function(o) {
            return o.name ==  option.name;
        });

        if(userTargetTool)
        {
            var value = "";
            if(option.type == "checkbox") {
                value = userTargetTool.value  == "on" ? option.arg : "";
            }
            else {
                value = option.arg + " " + userTargetTool.value;
            }

            // Update Params for tool
            var opt = { params: value }
            _.extend(option, opt);
        }
    });

    // add Files to to tool
    _.extend(toolListItem, userOptions.files);
    return toolListItem;
}

// This functions calls either basicParse or custom command defined in tooList
// Custom commands are called in t.parseCmd in toolList
// It lets you decide how to turn form data into command line parameter
// for the program call.
// Note: JS eval seems to play funny with nodemon.
function tempInsertOptions( toolList, toolOptions) {

    _.forEach( toolList, function(t){
        var targetToolOptions = getJsonTool( toolOptions.tools, t.progName );

        if( targetToolOptions )
        {
            var cmd = t.parseCmd || "basicParse";
            var result = "";
            try{
                result = eval(cmd)(t,targetToolOptions);
            }
            catch(err){
                Logger.error("Failed to call basic parse");
                Logger.error("CMD->", cmd, " has errored -> ");
                Logger.error("tool was: ", t );
                Logger.error("TargetToolOption was ", targetToolOptions);
                Logger.error("Error-> ", err);
                Logger.error("******************");
            }
        }

    });
    return toolList;
}

// This function takes the program name, the default parameters and user specified ones and creates 
// A command line call.
function createToolProgramCall ( toolListItem, files, options )
{
    var call = toolListItem.progName;
    var args = [ call, toolListItem.defaultArg ];

    _.forEach( toolListItem.options, function(o) {
        if( options  && options.prefixArg ) {
            args.push( o.arg );
        }
        args.push( o.params );
    });

    args.push(toolListItem.fileArgs);
    
   
    var filenames = _.map(files, 'filename' );
    var fullPath = _.union(args, filenames );
    var result = _.join( fullPath, " ");
    return result;
}

// Reduces the side of the object from the full job to a smaller version.
// Also creates a run command for the job.

function buildJobs( fullJobs, files, options ) {

    // Create an array of jobs with just the above mentioned keys, most important for passing.
    // Create a new property of the job that is the complete run call
    // TODO: Might eventually change this based on cmdType restType and cmdType
    var keys = [ 'displayName', 'progName', 'runType', 'defaultArg', 'fileArgs', 'options'];
 
    var halfJobs = _.map( fullJobs, obj => _.pick(obj, keys) );

    var jobs = _.map( halfJobs, obj => {
        obj['runCmd'] = createToolProgramCall(obj,files,options);
        return obj;
    });
    return jobs;
}

//Display key elements of a single tool object.
// TODO: update this used logger.
function displayTool( tool ) {

    var keys = [ 'displayName', 'progName', 'runType', 'defaultArg'];
    var t = _.pick(tool, keys);

    console.log("Start tool display");
    console.log(t);
    _.forEach( tool.options, function(o){
        console.log("Next Option");
        console.log("\t",o.displayName);
        console.log("\t",o.name);
        console.log("\tARG:", o.arg);
        console.log("\tPARAM:", o.params);
    });
}

// Generic all for all tools to be displayed.
function displayTools( tools ) {
    _.forEach( tools, displayTool );
}


// Exports below
module.exports.createJobRequests = createJobRequests;