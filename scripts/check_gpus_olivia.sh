#!/bin/bash
#
# check_gpus.sh - Check GPU availability on Olivia HPC (NRIS)
#
# Usage:
#   ./check_gpus.sh              # Show all GPU nodes
#   ./check_gpus.sh h200         # Show only H200 nodes
#   ./check_gpus.sh --free       # Show only nodes with free GPUs
#   ./check_gpus.sh -w           # Watch mode (updates every 5s)
#

# Olivia HPC configuration
PARTITION="accel"
ACCOUNT="nn12068k"

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# GH200 has 120GB GPU memory (96GB HBM3 + unified memory)
DEFAULT_GPU_MEM=120

# Parse arguments
FILTER_TYPE=""
ONLY_FREE=false
WATCH_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        --free)
            ONLY_FREE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [GPU_TYPE]"
            echo ""
            echo "Options:"
            echo "  -w, --watch     Watch mode (updates every 5 seconds)"
            echo "  --free          Show only nodes with free GPUs"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "GPU_TYPE: Filter by GPU type (h200, gh200)"
            echo ""
            echo "Examples:"
            echo "  $0              # Show all GPU nodes"
            echo "  $0 h200         # Show only H200 nodes"
            echo "  $0 --free       # Show only nodes with free GPUs"
            echo "  $0 -w           # Watch mode"
            exit 0
            ;;
        *)
            FILTER_TYPE="$1"
            shift
            ;;
    esac
done

show_gpu_status() {
    # Only clear in watch mode
    if [[ "$WATCH_MODE" == true ]]; then
        clear 2>/dev/null || true
    fi

    echo -e "${BOLD}${CYAN}=== OLIVIA HPC - GPU NODE STATUS ===${NC}"
    echo -e "${BOLD}Partition: ${PARTITION} | Account: ${ACCOUNT} | User: $USER${NC}"
    echo ""

    printf "${BOLD}%-15s | %-8s | %5s | %-18s | %8s | %9s | %8s | %-12s | %s${NC}\n" \
           "Node" "GPU" "Total" "Status Breakdown" "GPU Mem" "CPUs Free" "Mem Free" "State" "Users"
    echo "----------------|----------|-------|--------------------| ---------|-----------|----------|--------------|----------------"

    # Get list of GPU nodes
    nodes=$(sinfo -p ${PARTITION} -N -h -o "%N" | sort -u)
    
    total_gpus=0
    total_used=0
    total_free=0
    total_idle=0
    total_reserved=0
    node_count=0
    
    while IFS= read -r node; do
        # Get node details
        node_info=$(scontrol show node "$node" 2>/dev/null)
        
        if [[ -z "$node_info" ]]; then
            continue
        fi
        
        # Extract GPU information
        if [[ $node_info =~ Gres=gpu:([^:]+):([0-9]+) ]]; then
            gpu_type="${BASH_REMATCH[1]}"
            gpu_total="${BASH_REMATCH[2]}"
        else
            continue
        fi
        
        # Filter by GPU type if specified
        if [[ -n "$FILTER_TYPE" ]] && [[ ! "$gpu_type" =~ $FILTER_TYPE ]]; then
            continue
        fi
        
        # Extract allocated GPUs
        gpu_used=0
        if [[ $node_info =~ AllocTRES=.*gres/gpu[^=]*=([0-9]+) ]]; then
            gpu_used="${BASH_REMATCH[1]}"
        fi
        
        # Calculate free GPUs
        gpu_free=$((gpu_total - gpu_used))
        
        # Filter only free if specified
        if [[ "$ONLY_FREE" == true ]] && [[ $gpu_free -eq 0 ]]; then
            continue
        fi
        
        # Extract GPU memory dynamically from SLURM features
        gpu_mem_per_unit=0

        # Extract features from node info (look for AvailableFeatures or Features)
        features=""
        if [[ $node_info =~ AvailableFeatures=([^ ]+) ]]; then
            features="${BASH_REMATCH[1]}"
        elif [[ $node_info =~ Features=([^ ]+) ]]; then
            features="${BASH_REMATCH[1]}"
        fi

        # Parse GPU memory from features (e.g., gpu16g, gpu32g, gpu40g, gpu80g)
        if [[ $features =~ gpu([0-9]+)g ]]; then
            gpu_mem_per_unit="${BASH_REMATCH[1]}"
        fi

        # Format GPU memory string (default to GH200's 120GB if not found)
        if [[ $gpu_mem_per_unit -gt 0 ]]; then
            gpu_mem_str="${gpu_mem_per_unit}G"
        elif [[ "$gpu_type" == "h200" ]] || [[ "$gpu_type" == "gh200" ]]; then
            gpu_mem_str="${DEFAULT_GPU_MEM}G"
        else
            gpu_mem_str="N/A"
        fi
        
        # Extract CPU information
        if [[ $node_info =~ CPUAlloc=([0-9]+) ]]; then
            cpu_alloc="${BASH_REMATCH[1]}"
        else
            cpu_alloc=0
        fi
        
        if [[ $node_info =~ CPUTot=([0-9]+) ]]; then
            cpu_tot="${BASH_REMATCH[1]}"
        else
            cpu_tot=0
        fi
        cpu_free=$((cpu_tot - cpu_alloc))
        
        # Extract memory information
        if [[ $node_info =~ RealMemory=([0-9]+) ]]; then
            mem_tot="${BASH_REMATCH[1]}"
        else
            mem_tot=0
        fi
        
        if [[ $node_info =~ AllocMem=([0-9]+) ]]; then
            mem_alloc="${BASH_REMATCH[1]}"
        else
            mem_alloc=0
        fi
        mem_free=$(( (mem_tot - mem_alloc) / 1024 ))  # Convert to GB
        
        # Extract state
        if [[ $node_info =~ State=([A-Z+]+) ]]; then
            state="${BASH_REMATCH[1]}"
        else
            state="UNKNOWN"
        fi

        # Determine GPU status breakdown based on node state
        # States: IDLE, MIXED, ALLOCATED, RESERVED, PLANNED, DRAIN, DOWN
        gpu_idle=0
        gpu_reserved=0
        gpu_allocated=$gpu_used

        # Check if node is in reserved state
        if [[ "$state" == *"RESERVED"* ]] || [[ "$state" == *"RESV"* ]]; then
            # All free GPUs are reserved
            gpu_reserved=$gpu_free
            gpu_idle=0
        # Check if node is in planned state
        elif [[ "$state" == *"PLANNED"* ]] || [[ "$state" == *"PLND"* ]]; then
            # All free GPUs are planned/reserved
            gpu_reserved=$gpu_free
            gpu_idle=0
        else
            # Free GPUs are actually idle
            gpu_idle=$gpu_free
            gpu_reserved=0
        fi

        # Build status breakdown string
        status_parts=()
        if [[ $gpu_idle -gt 0 ]]; then
            status_parts+=("${gpu_idle} idle")
        fi
        if [[ $gpu_allocated -gt 0 ]]; then
            status_parts+=("${gpu_allocated} used")
        fi
        if [[ $gpu_reserved -gt 0 ]]; then
            status_parts+=("${gpu_reserved} resv")
        fi

        # Join with commas
        status_breakdown=$(IFS=', '; echo "${status_parts[*]}")
        if [[ -z "$status_breakdown" ]]; then
            status_breakdown="-"
        fi
        
        # Get users running on this node
        users=$(squeue -h -w "$node" -o "%u" | sort -u | tr '\n' ',' | sed 's/,$//' | sed 's/,/, /g')
        if [[ -z "$users" ]]; then
            users="-"
        fi
        
        # Color coding based on availability (use idle, not free, for accurate coloring)
        if [[ $gpu_idle -eq 0 ]]; then
            color=$RED
        elif [[ $gpu_idle -eq $gpu_total ]]; then
            color=$GREEN
        else
            color=$YELLOW
        fi
        
        # Print node information
        printf "${color}%-15s | %-8s | %5d | %-18s | %8s | %9d | %6dG | %-12s | %s${NC}\n" \
               "$node" "$gpu_type" "$gpu_total" "$status_breakdown" \
               "$gpu_mem_str" "$cpu_free" "$mem_free" "$state" "$users"
        
        # Update totals
        total_gpus=$((total_gpus + gpu_total))
        total_used=$((total_used + gpu_used))
        total_free=$((total_free + gpu_free))
        total_idle=$((total_idle + gpu_idle))
        total_reserved=$((total_reserved + gpu_reserved))
        node_count=$((node_count + 1))
    done <<< "$nodes"
    
    echo "--------------------------------------------------------------------------------------------------------------"

    # Build total status breakdown
    total_status_parts=()
    if [[ $total_idle -gt 0 ]]; then
        total_status_parts+=("${total_idle} idle")
    fi
    if [[ $total_used -gt 0 ]]; then
        total_status_parts+=("${total_used} used")
    fi
    if [[ $total_reserved -gt 0 ]]; then
        total_status_parts+=("${total_reserved} resv")
    fi
    total_status=$(IFS=', '; echo "${total_status_parts[*]}")

    printf "${BOLD}%-15s | %-8s | %5d | %-18s | %8s | %9s | %8s | %-12s | %s${NC}\n" \
           "TOTAL" "($node_count)" "$total_gpus" "$total_status" "" "" "" "" ""
    
    echo ""
    echo -e "${CYAN}Legend:${NC}"
    echo -e "  ${GREEN}Green${NC}  = All GPUs free"
    echo -e "  ${YELLOW}Yellow${NC} = Partially allocated"
    echo -e "  ${RED}Red${NC}    = Fully allocated"
    
    if [[ "$WATCH_MODE" == true ]]; then
        echo ""
        echo -e "${BLUE}Press Ctrl+C to exit watch mode${NC}"
    fi
}

# Main execution
if [[ "$WATCH_MODE" == true ]]; then
    while true; do
        show_gpu_status
        sleep 5
    done
else
    show_gpu_status
fi