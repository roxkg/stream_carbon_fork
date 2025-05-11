from zigzag.datatypes import LayerOperand

from stream.cost_model.scheduler import schedule_graph
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.carbonparam import CarbonParam
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import plot_timeline_brokenaxes
from stream.workload.onnx_workload import ComputationNodeWorkload


class StreamCostModelEvaluation:
    """Stream's cost model evaluation class which includes a scheduler and memory utilization tracer.
    Throughout SCME will be used as abbreviation.
    This evaluation computes the total latency and activation memory utilization throughout the inference.
    """

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        carbon_param: CarbonParam,
        embodied_carbon: float,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
    ) -> None:
        # Initialize the SCME by setting the workload graph to be scheduled
        self.workload = workload
        self.accelerator = accelerator
        self.carbon_param = carbon_param
        self.embodied_carbon = carbon_param.cemb
        self.energy: float | None = None
        self.total_cn_onchip_energy: float | None = None
        self.total_cn_offchip_link_energy: float | None = None
        self.total_cn_offchip_memory_energy: float | None = None
        self.total_eviction_to_offchip_link_energy: float | None = None
        self.total_eviction_to_offchip_memory_energy: float | None = None
        self.total_sink_layer_output_offchip_link_energy: float | None = None
        self.total_sink_layer_output_offchip_memory_energy: float | None = None
        self.total_core_to_core_link_energy: float | None = None
        self.total_core_to_core_memory_energy: float | None = None

        self.latency: int | None = None
        self.carbon: float | None = None
        self.CD: float | None = None
        self.ED: float | None = None
        self.tCDP: float | None = None
        self.area_total: int | None = None
        self.max_memory_usage = None
        self.core_timesteps_delta_cumsums = None
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

    def __str__(self):
        return f"SCME(energy={self.energy:.2e}, latency={self.latency:.2e}, carbon={self.carbon:.2e})"

    def run(self):
        """Run the SCME by scheduling the graph through time.
        The scheduler takes into account inter-core data movement and also tracks energy and memory through the memory
        manager.
        This assumes each node in the graph has an energy and runtime of the core to which they are allocated to.
        """
        results = schedule_graph(
            self.workload,
            self.accelerator,
            operands_to_prefetch=self.operands_to_prefetch,
            scheduling_order=self.scheduling_order,
        )
        self.latency = results[0]
        self.total_cn_onchip_energy = results[1]
        self.total_cn_offchip_link_energy = results[2]
        self.total_cn_offchip_memory_energy = results[3]
        self.total_eviction_to_offchip_link_energy = results[4]
        self.total_eviction_to_offchip_memory_energy = results[5]
        self.total_sink_layer_output_offchip_link_energy = results[6]
        self.total_sink_layer_output_offchip_memory_energy = results[7]
        self.total_core_to_core_link_energy = results[8]
        self.total_core_to_core_memory_energy = results[9]
        self.schedule_trace = results[-1]

        f = open(f"outputs/schedule_trace.txt", "w")
        f.write(f"{self.schedule_trace}\n")
        f.close()

        self.energy = (
            self.total_cn_onchip_energy
            + self.total_cn_offchip_link_energy
            + self.total_cn_offchip_memory_energy
            + self.total_eviction_to_offchip_link_energy
            + self.total_eviction_to_offchip_memory_energy
            + self.total_sink_layer_output_offchip_link_energy
            + self.total_sink_layer_output_offchip_memory_energy
            + self.total_core_to_core_link_energy
            + self.total_core_to_core_memory_energy
        )

        lifespan = self.carbon_param.lifetime
        frequency = self.carbon_param.frequency
        taskspan = self.latency/(frequency*10**9)/3600
        energy = self.energy/(10**12)/3600000
        power = energy/taskspan
        
        self.carbon = lifespan / taskspan * energy * self.carbon_param.CI_op
        # self.carbon = energy * self.carbon_param.CI_op
        self.CD = sum(self.embodied_carbon.values()) /lifespan * taskspan * taskspan * 3600 * 1000
        # self.CD = sum(self.embodied_carbon.values()) * taskspan * 3600 * 1000
        self.CD = float(self.CD)
        # self.ED = self.energy/(10**12) * taskspan * 3600
        self.ED = self.carbon_param.CI_op * power * taskspan*taskspan * 3600   # g*s
        # self.ED = self.carbon_param.CI_op * power * lifespan*taskspan * 3600   # g*s
        # self.tCDP = self.CD + (self.carbon_param.lifetime * self.carbon_param.CI_op * power) * taskspan * 3600
        self.tCDP = self.ED + self.CD
        # self.tCDP,self.CD,self.ED = self.compute_tcdp_per_core()
        # self.area_total, self.mem_area = self.collect_area_data()

    """
    def collect_area_data(self):
        area_total = 0
        # get mem area
        mem_area = 0
        mem_area_breakdown: dict[str, float] = {}
        for core in self.accelerator.core_list:
            with open("outputs/area_log.txt", "a") as file: 
                file.write(f"current core id = {core.id}, with unit count = {core.operational_array.total_unit_count}\n")
            area_total += core.operational_array.total_unit_count*1
            for mem in core.memory_hierarchy.mem_level_list:
            # for mem in self.accelerator.mem_level_list:
                memory_instance = mem.memory_instance
                memory_instance_name = memory_instance.name
                mem_area += memory_instance.area
                mem_area_breakdown[memory_instance_name] = memory_instance.area
                with open("outputs/area_mem_log.txt", "a") as file: 
                    file.write(f"memory_instance_name = {memory_instance_name}, with instance area = {memory_instance.area}\n")
            # get total area
            area_total += mem_area
        return area_total, mem_area
    """

    def compute_tcdp_per_core(self):
        CI = self.carbon_param.CI_op
        tcdp_core = {}
        tcdp_total = 0
        CD_total= 0
        ED_total= 0

        for core_id, events in self.schedule_trace.items():
            # 获取 core 的制造碳排
            # core_obj = self.accelerator.get_core(core_id)
            cem_core = self.embodied_carbon.get(core_id)
            if cem_core is None:
                assert f"Core {core_id} not found in embodied carbon data."
                cem_core = 0  # fallback

            # 分离 idle 和 active 时间段
            active_events = [e for e in events if e["type"] == "active"]
            idle_events = [e for e in events if e["type"] == "idle"]

            # active 总运行时间
            active_time = sum(e["latency"] for e in active_events)
            # 总运行跨度（包括 idle）
            if active_events:
                span_time = max(e["end"] for e in active_events) - min(e["start"] for e in active_events)
            else:
                span_time = 0

            # 使用哪个作为 latency
            d_core = active_time  # if use_active_time else span_time

            # operational carbon（每个 node 的 energy × CI）
            cop_core = sum(
                e["node_obj"].get_onchip_energy() + e["node_obj"].get_offchip_energy()
                for e in active_events
            ) * CI /(10**12) / 3600000

            # tCDP = (Cem + Cop) × D
            d_core = active_time / (self.carbon_param.frequency*10**9)
            CD =  cem_core * d_core/(self.carbon_param.lifetime * 3600) *d_core * 1000
            ED = cop_core * d_core
            tcdp = CD + ED 
            CD_total += CD
            ED_total += ED
            tcdp_total += tcdp
        total_link_energy = (
            self.total_cn_offchip_link_energy +
            self.total_eviction_to_offchip_link_energy +
            self.total_sink_layer_output_offchip_link_energy +
            self.total_core_to_core_link_energy
        )

        total_memory_energy = (
            self.total_cn_offchip_memory_energy +
            self.total_eviction_to_offchip_memory_energy +
            self.total_sink_layer_output_offchip_memory_energy +
            self.total_core_to_core_memory_energy
        )
        tcdp_total += self.latency/ (self.carbon_param.frequency*10**9) * (total_link_energy + total_memory_energy) * CI /(10**12) / 3600000
        ED_total += self.latency/ (self.carbon_param.frequency*10**9) * (total_link_energy + total_memory_energy) * CI /(10**12) / 3600000
        return tcdp_total, CD_total, ED_total


    def collect_area_data(self): 
        area_total = 0 
        noc_area = 0
        core_area = 0
        for core in self.accelerator.core_list: 
            core_area += core.core_area
            noc_area += core.noc_area
            area_total = area_total + core_area + noc_area
        return area_total, noc_area, core_area

    def plot_schedule(
        self,
        plot_full_schedule: bool = False,
        draw_dependencies: bool = True,
        plot_data_transfer: bool = False,
        section_start_percent: tuple[int, ...] = (0, 50, 95),
        percent_shown: tuple[int, ...] = (5, 5, 5),
        fig_path: str = "outputs/schedule_plot.png",
    ):
        """Plot the schedule of this SCME."""
        if plot_full_schedule:
            section_start_percent = (0,)
            percent_shown = (100,)
        plot_timeline_brokenaxes(
            self,
            draw_dependencies,
            section_start_percent,
            percent_shown,
            plot_data_transfer,
            fig_path,
        )

    def plot_memory_usage(self, fig_path: str = "outputs/memory_usage_plot.png"):
        """Plot the memory usage of this SCME."""
        plot_memory_usage(self.accelerator.memory_manager, fig_path)
