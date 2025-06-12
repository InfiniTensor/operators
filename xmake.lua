add_rules("mode.debug", "mode.release")
-- Define color codes
local GREEN = '\27[0;32m'
local YELLOW = '\27[1;33m'
local NC = '\27[0m'  -- No Color

add_includedirs("include")

option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable cpu kernel")
    add_defines("ENABLE_CPU")
option_end()

option("omp")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable OpenMP support for cpu kernel")
option_end()

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Nvidia GPU kernel")
    add_defines("ENABLE_NV_GPU")
option_end()

option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Cambricon MLU kernel")
    add_defines("ENABLE_CAMBRICON_MLU")
option_end()

option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Ascend NPU kernel")
    add_defines("ENABLE_ASCEND_NPU")
option_end()

option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Metax GPU kernel")
    add_defines("ENABLE_METAX_GPU")
option_end()


option("mthreads-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable MThreads GPU kernel")
    add_defines("ENABLE_MTHREADS_GPU")
option_end()

option("sugon-dcu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Sugon DCU kernel")
    add_defines("ENABLE_SUGON_DCU")
    add_defines("ENABLE_NV_GPU")
option_end()

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end

if has_config("cpu") then

    add_defines("ENABLE_CPU")
    target("cpu")
        on_install(function (target) end)
        set_kind("static")

        if not is_plat("windows") then
            add_cxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cpu/*.cc", "src/ops/*/cpu/*.cc")
        if has_config("omp") then
            add_cxflags("-fopenmp")
            add_ldflags("-fopenmp")
        end
    target_end()

end

if has_config("nv-gpu", "sugon-dcu") then
    add_defines("ENABLE_NV_GPU")
    if has_config("sugon-dcu") then
        add_defines("ENABLE_SUGON_DCU")
    end
    local CUDA_ROOT = os.getenv("CUDA_ROOT") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
    if CUDA_ROOT ~= nil then
        add_includedirs(CUDA_ROOT .. "/include")
    end
    if CUDNN_ROOT ~= nil then
        add_includedirs(CUDNN_ROOT .. "/include")
    end

    target("nv-gpu")
        set_kind("static")
        on_install(function (target) end)
        set_policy("build.cuda.devlink", true)

        set_toolchains("cuda")
        add_links("cublas")
        add_links("cudnn")
        add_cugencodes("native")

        if is_plat("windows") then
            add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
            if CUDNN_ROOT ~= nil then
                add_linkdirs(CUDNN_ROOT .. "\\lib\\x64")
            end
        else
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
            add_cxxflags("-fPIC")
        end

        set_languages("cxx17")
        add_files("src/devices/cuda/*.cc", "src/ops/*/cuda/*.cu")
        add_files("src/ops/*/cuda/*.cc")
    target_end()

end

if has_config("cambricon-mlu") then

    add_defines("ENABLE_CAMBRICON_MLU")
    add_includedirs("/usr/local/neuware/include")
    add_linkdirs("/usr/local/neuware/lib64")
    add_linkdirs("/usr/local/neuware/lib")
    add_links("libcnrt.so")
    add_links("libcnnl.so")
    add_links("libcnnl_extra.so")
    add_links("libcnpapi.so")

    rule("mlu")
        set_extensions(".mlu")

        on_load(function (target)
            target:add("includedirs", "include")
        end)

        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))

            local cc = "/usr/local/neuware/bin/cncc"

            local includedirs = table.concat(target:get("includedirs"), " ")
            local args = {"-c", sourcefile, "-o", objectfile, "-I/usr/local/neuware/include", "--bang-mlu-arch=mtp_613", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"}

            for _, includedir in ipairs(target:get("includedirs")) do
                table.insert(args, "-I" .. includedir)
            end

            os.execv(cc, args)
            table.insert(target:objectfiles(), objectfile)
        end)

    rule_end()

    target("cambricon-mlu")
        set_kind("static")
        on_install(function (target) end)
        set_languages("cxx17")
        add_files("src/devices/bang/*.cc", "src/ops/*/bang/*.cc")
        add_files("src/ops/*/bang/*.mlu", {rule = "mlu"})
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")
    target_end()

end

if has_config("mthreads-gpu") then

    add_defines("ENABLE_MTHREADS_GPU")
    local musa_home = os.getenv("MUSA_INSTALL_PATH")
    -- Add include dirs
    add_includedirs(musa_home .. "/include")
    -- Add shared lib
    add_linkdirs(musa_home .. "/lib")
    add_links("libmusa.so")
    add_links("libmusart.so")
    add_links("libmudnn.so")
    add_links("libmublas.so")

    rule("mu")
        set_extensions(".mu")
        on_load(function (target)
            target:add("includedirs", "include")
        end)

        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))

            local mcc = "/usr/local/musa/bin/mcc"
            local includedirs = table.concat(target:get("includedirs"), " ")
            local args = {"-c", sourcefile, "-o", objectfile, "-I/usr/local/musa/include", "-O3", "-fPIC", "-Wall", "-std=c++17", "-pthread"}
            for _, includedir in ipairs(target:get("includedirs")) do
                table.insert(args, "-I" .. includedir)
            end

            os.execv(mcc, args)
            table.insert(target:objectfiles(), objectfile)
        end)
    rule_end()

    target("mthreads-gpu")
        set_kind("static")
        set_languages("cxx17")
        add_files("src/devices/musa/*.cc", "src/ops/*/musa/*.cc")
        add_files("src/ops/*/musa/*.mu", {rule = "mu"})
        add_cxflags("-lstdc++ -Wall -fPIC")
    target_end()

end

if has_config("ascend-npu") then

    add_defines("ENABLE_ASCEND_NPU")
    local ASCEND_HOME = os.getenv("ASCEND_HOME")
    local SOC_VERSION = os.getenv("SOC_VERSION")

    -- Add include dirs
    add_includedirs(ASCEND_HOME .. "/include")
    add_includedirs(ASCEND_HOME .. "/include/aclnn")
    add_linkdirs(ASCEND_HOME .. "/lib64")
    add_links("libascendcl.so")
    add_links("libnnopbase.so")
    add_links("libopapi.so")
    add_links("libruntime.so")
    add_linkdirs(ASCEND_HOME .. "/../../driver/lib64/driver")
    add_links("libascend_hal.so")
    local builddir = string.format(
            "%s/build/%s/%s/%s",
            os.projectdir(),
            get_config("plat"),
            get_config("arch"),
            get_config("mode")
        )
    rule("ascend-kernels")
        before_link(function ()
            local ascend_build_dir = path.join(os.projectdir(), "src/devices/ascend")
            os.cd(ascend_build_dir)
            os.exec("make")
            os.exec("cp $(projectdir)/src/devices/ascend/build/lib/libascend_kernels.a "..builddir.."/")
            os.cd(os.projectdir())

        end)
        after_clean(function ()
            local ascend_build_dir = path.join(os.projectdir(), "src/devices/ascend")
            os.cd(ascend_build_dir)
            os.exec("make clean")
            os.cd(os.projectdir())
            os.rm(builddir.. "/libascend_kernels.a")

        end)
    rule_end()

    target("ascend-npu")
        -- Other configs
        set_kind("static")
        set_languages("cxx17")
        on_install(function (target) end)
        -- Add files
        add_files("src/devices/ascend/*.cc", "src/ops/*/ascend/*.cc")
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")

        -- Add operator
        add_rules("ascend-kernels")
        add_links(builddir.."/libascend_kernels.a")

    target_end()
end

if has_config("metax-gpu") then

    add_defines("ENABLE_METAX_GPU")
    local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")

    add_includedirs(MACA_ROOT .. "/include")
    add_linkdirs(MACA_ROOT .. "/lib")
    -- add_linkdirs(MACA_ROOT .. "htgpu_llvm/lib")
    add_links("libhcdnn.so")
    add_links("libhcblas.so")
    add_links("libhcruntime.so")

    rule("maca")
        set_extensions(".maca")

        on_load(function (target)
            target:add("includedirs", "include")
        end)

        on_build_file(function (target, sourcefile)
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))
            local htcc = "/opt/hpcc/htgpu_llvm/bin/htcc"

            local includedirs = table.concat(target:get("includedirs"), " ")
            local args = { "-x", "hpcc", "-c", sourcefile, "-o", objectfile, "-I/opt/hpcc/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}

            for _, includedir in ipairs(target:get("includedirs")) do
                table.insert(args, "-I" .. includedir)
            end

            os.execv(htcc, args)
            table.insert(target:objectfiles(), objectfile)
        end)
    rule_end()

    target("metax-gpu")
        set_kind("static")
        on_install(function (target) end)
        set_languages("cxx17")
        add_files("src/devices/maca/*.cc", "src/ops/*/maca/*.cc")
        add_files("src/ops/*/maca/*.maca", {rule = "maca"})
        add_cxflags("-lstdc++ -Werror -fPIC")
    target_end()

end


toolchain("sugon-dcu-linker")
    set_toolset("sh", "nvcc")
toolchain_end()

target("infiniop")
    set_kind("shared")

    if has_config("cpu") then
        add_deps("cpu")
    end
    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end
    if has_config("sugon-dcu") then
        local builddir = string.format(
            "build/%s/%s/%s",
            get_config("plat"),
            get_config("arch"),
            get_config("mode")
        )
        add_shflags("-s", "-shared", "-fPIC")
        add_links("cublas", "cudnn", "cudadevrt", "cudart_static", "rt", "pthread", "dl")
        -- Using -lnv-gpu will fail, manually link the target using full path
        add_deps("nv-gpu", {inherit = false})
        add_links(builddir.."/libnv-gpu.a")
        set_toolchains("sugon-dcu-linker")
    end

    if has_config("cambricon-mlu") then
        add_deps("cambricon-mlu")
    end
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end
    if has_config("metax-gpu") then
        add_deps("metax-gpu")
    end
    if has_config("mthreads-gpu") then
        add_deps("mthreads-gpu")
    end
    set_languages("cxx17")
    add_files("src/devices/handle.cc")
    add_files("src/ops/*/operator.cc")
    add_files("src/tensor/*.cc")
    after_build(function (target) print(YELLOW .. "You can install the libraries with \"xmake install\"" .. NC) end)

    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
    add_installfiles("include/(**/*.h)", {prefixdir = "include"})
    add_installfiles("include/*.h", {prefixdir = "include"})

target_end()
