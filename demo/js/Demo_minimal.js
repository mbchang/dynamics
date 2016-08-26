/**
* The Matter.js demo page controller and example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @class Demo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.test(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');

    var Demo = {};
    Matter.Demo = Demo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile')
        var CircularJSON = require('circular-json')
        var assert = require('assert')
        var utils = require('../../utils')
        // var sleep = require('sleep')
        var PImage = require('pureimage');
        var fs = require('fs');
        require('./Examples')
        module.exports = Demo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Runner = Matter.Runner,
        Render = Matter.Render;

    // Create the engine
    Demo.run = function(json_data) {

        //TODO: note that here you should load the demo engine with the json file

        // load the config file here.
        console.log(json_data)
        let data = json_data.trajectories
        let config = json_data.config
        console.log(config)

        // let data = json_data

        var demo = {}
        demo.offset = 5;  // world offset
        demo.config = {}
        demo.config.cx = 400;
        demo.config.cy = 300;
        demo.config.masses = [1, 5, 25]
        demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        demo.config.sizes = [2/3, 1, 3/2]  // multiples
        demo.config.drastic_sizes = [1/2, 1, 2]  // multiples  NOTE THAT WE HAVE THREE VALUES NOW!
        // demo.config.drastic_sizes = [1/2, 2]  // multiples  NOTE THAT WE HAVE THREE VALUES NOW!
        demo.config.object_base_size = {'ball': 60, 'obstacle': 80, 'block': 20 }  // radius of ball, side of square obstacle, long side of block
        demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        demo.config.g = 0 // default? [0,1] Or should we make this a list? The index of the one hot. 0 is no, 1 is yes
        demo.config.f = 0 // default? [0,1]
        demo.config.p = 0 // default? [0,1,2]
        demo.config.max_velocity = 60

        demo.cx = demo.config.cx;
        demo.cy = demo.config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        demo.engine = Engine.create()
        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                    max: { x: demo.width, y: demo.height }}


        // here let's put a isBrowser condition
        if (_isBrowser) {  // do everything normally.
            demo.runner = Engine.run(demo.engine)
            demo.container = document.getElementById('canvas-container');
            demo.render = Render.create({element: demo.container, engine: demo.engine, 
                                        hasBounds: true, options:{height:demo.height, width:demo.width}})
            Render.run(demo.render)
        } else {
            // run the engine
            demo.runner = Runner.create()
            demo.runner.isFixed = true
            var pcanvas = PImage.make(demo.width, demo.height);  // 693
            pcanvas.style = {}  
            console.log(pcanvas)
            demo.render = Render.create({
                element: 17, // dummy
                canvas: pcanvas,
                engine: demo.engine,
            })
            

            // Events.on(demo.render, 'afterRender', function(e) {
            //     var img = demo.render.canvas.toDataURL("image"+e.timestamp+".png")
            //     var data = img.replace(/^data:image\/\w+;base64,/, "");
            //     // var buf = new Buffer(data, 'base64');
            //     // fs.writeFile('image.png', buf);
            //     // document.write('<img src="'+img+'"/>');  // how do I save this?
            //     // PImage.encodePNG(img, fs.createWriteStream('out.png'), function(err) {
            //     // console.log("wrote out the png file to out.png");
            //     // });
            // });

            demo.render.hasBounds = true
            demo.render.options.height = demo.height
            demo.render.options.width = demo.width
            demo.render.canvas.height = demo.height
            demo.render.canvas.width = demo.width
        }


        if (demo.render) {
            var renderOptions = demo.render.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = false;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.background = '#fff';
        }

        var mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}// TODO eventually call Example[config.env].mass_colors

        // Ok, now let's manually update
        if (_isBrowser) {
            Runner.stop(demo.runner) // seems like this is causing the problem!
        }

        var trajectories = data[0]  // extra 0 for batch mode
        var num_obj = trajectories.length
        var num_steps = trajectories[0].length
        config.trajectories = trajectories

        Example[config.env](demo, config)  // here you have to assign balls initial positions according to the initial timestep of trajectories.

        // here you have to 
        console.log(config)

        console.log(num_steps)
        if (config.env=='tower') {
            var i = 2  // if I set i to < 2 then I get very weird behavior
        } else {
            var i = 0
        }

        function f() {
            console.log( 'i', i );
            var entities = Composite.allBodies(demo.engine.world)
                .filter(function(elem) {
                            return elem.label === 'Entity';
                        })
            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            for (id = 0; id < entity_ids.length; id++) { //id = 0 corresponds to world!
                var body = Composite.get(demo.engine.world, entity_ids[id], 'body')
                // set the position here
                if (i < config.num_past) {
                    body.render.strokeStyle = '#FFA500'// orange #551A8B is purple
                } else {
                    body.render.strokeStyle = '#551A8B'// orange #551A8B is purple
                }
                body.render.lineWidth = 5

                console.log('set position')

                Body.setPosition(body, trajectories[id][i].position)
                Body.setAngle(body, trajectories[id][i].angle)
                console.log(id, trajectories[id][i].position, trajectories[id][i].velocity, trajectories[id][i].angle)

            }

            Runner.tick(demo.runner, demo.engine);


            if (!_isBrowser) {
                demo.render.context.fillStyle = 'white'
                demo.render.context.fillRect(0,0,demo.width,demo.height)
                Render.world(demo.render)
                let filename = 'out'+i+'_'+i+'.png'  // TODO! rename
                PImage.encodePNG(demo.render.canvas, fs.createWriteStream(filename), function(err) {
                    console.log("wrote out the png file to "+filename);
                });

                // TODO: don't have to set timeout.
                // console.log('hi')
            }


            i++;
            if( i < num_steps ){
                if (_isBrowser) {
                    setTimeout( f, 5 );
                } else {
                    setTimeout( f, 0 );
                }
            }
        }
        f();


    }

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        console.log('a')
        window.loadFile = function loadFile(file){
            var fr = new FileReader();
            fr.onload = function(){
                Demo.run(window.CircularJSON.parse(fr.result))
            }
            fr.readAsText(file)
        }
    } else {
        // here load the json file here
        let loaded_json= jsonfile.readFileSync('gt_batch383.json')
        // console.log('b')
        // console.log(loaded_file)
        Demo.run(loaded_json)
    }
})();
