## DemoPlugin for ImFusionSuite

[ImFusionSuite](https://www.imfusion.com/products/imfusion-suite) provides high-performance visualization and processing for medical data, it is developed by [ImFusion GmbH](https://www.imfusion.com/).
It also enables to implement custom C++ plugins that make use of the provided SDK.

This repo provides a baseline code for the implementation of a custom plugin.

To build it you require:
- ImFusionSuite SDK
- Qt5

It can be easily build using CMake, specifing the ImFusionLib_DIR and Qt5_DIR vabiables, e.g. (from the repository root)

```bash
mkdir build && cd build
cmake -DImFusionLib_DIR=<PATH_TO_IMFUSIONLIB_CMAKE> -DQt5_DIR=<PATH_TO_QT5_CMAKE> ..
make
```

### Create your own plugin starting from this one!

You can easily create a new plugin for ImFusionSuite using this one as a starting point.
For instance, if you want to create a plugin called `MyAmazingPlugin`, these are the required steps:

- Copy the entire repo and change the root folde name to `MyAmazingPlugin`
- Rename the folder `include/demo_plugin` to `include/my_amazing_plugin`
- In every .hpp and .cpp file, rename the defined namespace to `namespace MyAmazingPlugin`

- In every .cpp file, modify the includes that use `demo_plugin/` to `my_amazing_plugin/`
- In factory.cpp, modify line 12 to `PluginAlgorithmFactory::PluginAlgorithmFactory() { registerAlgorithm<PluginAlgorithm>("MyCategory;My Amazing Plugin"); }`
    
- In plugin.cpp, modify line 10 to `return new ImFusion::MyAmazingPlugin::Plugin;`
      
- In CMakeLists.txt, modify the project name from `demo_plugin` to `my_amazing_plugin` (it has to match the folder name under the include folder)










### Contacts

[![TUM](http://campar.in.tum.de/files/goeblr/TUM_Web_Logo_blau.png "TUM Logo")](http://tum.de)        
[Chair for Computer Aided Medical Procedures](http://campar.in.tum.de/)        
[Technical University of Munich](www.tum.de), Germany.       

<b>Salvatore Virga</b> : [salvo.virga@tum.de][salvo_email]     

[salvo_email]: salvo.virga@tum.de